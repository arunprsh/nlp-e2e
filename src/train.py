from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
import tensorflow as tf
import argparse
import logging
import tarfile
import boto3
import sys
import os


s3_client = boto3.client('s3')
s3 = boto3.resource('s3')


def get_matching_s3_keys(bucket: str, prefix: str, suffix: str):
    kwargs = {'Bucket': bucket, 'Prefix': prefix}
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.endswith(suffix):
                yield key
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
             break
                
                
def get_file_paths(directory):
    file_paths = [] 
    for root, directories, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)  
    return file_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train-batch-size', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--learning_rate', type=str, default=5e-5)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--model_s3', type=str, default=None)

    # Data, model, and output directories
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
    # parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    # parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--n_gpus', type=str, default=os.environ['SM_NUM_GPUS'])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(level=logging.getLevelName('INFO'), 
                        handlers=[logging.StreamHandler(sys.stdout)], 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                       )

    # Load model and tokenizer
    model = TFAutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])

    # Pre-process train dataset
    train_dataset = train_dataset.map(lambda e: tokenizer(e['text'], 
                                                          truncation=True, 
                                                          padding='max_length'), 
                                      batched=True)
    
    train_dataset.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'label'])

    train_features = {
        x: train_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length])
        for x in ['input_ids', 'attention_mask']
    }
    
    tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_features, 
                                                           train_dataset['label'])
                                                         ).batch(args.train_batch_size)

    # Pre-process test dataset
    test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], 
                                                        truncation=True, 
                                                        padding='max_length'), 
                                    batched=True)
    
    test_dataset.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'label'])

    test_features = {
        x: test_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length])
        for x in ['input_ids', 'attention_mask']
    }
    
    tf_test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_dataset['label'])
                                                        ).batch(args.eval_batch_size)

    # Set optimizer, loss and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Training
    if args.do_train:
        train_results = model.fit(tf_train_dataset, epochs=args.epochs, batch_size=args.train_batch_size)
        logger.info('-------------------- Train --------------------')
        output_eval_file = os.path.join(args.output_dir, 'train_results.txt')
        with open(output_eval_file, 'w') as writer:
            logger.info(train_results)
            for key, value in train_results.history.items(): 
                logger.info(f'{key} = {value}')
                writer.write(f'{key} = {value}')
    
    """
    # Evaluation
    if args.do_eval:
        result = model.evaluate(tf_test_dataset, batch_size=args.eval_batch_size, return_dict=True)
        logger.info('-------------------- Evaluation --------------------')
        output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
        with open(output_eval_file, 'w') as writer:
            logger.info(result)
            for key, value in result.items():
                logger.info(f'{key} = {value}')
                writer.write(f'{key} = {value}')
    """
    
    # Save result
    model.save_pretrained(args.output_dir)
    
    model_out = args.model_s3

    _, uri = model_out.split('//')
    bucket, prefix = uri.split('/')[0], uri.split('/')[1:]
    prefix = '/'.join(prefix) + '/'
    
    tar = tarfile.open(f'{args.output_dir}/model.tar.gz', 'w:gz')
    
    
    file_paths = get_file_paths(args.output_dir)
    for file_path in file_paths:
        file_ = file_path.split('/')[-1]
        if file_.endswith('h5') or file_.endswith('json'):
            tar.add(file_path, arcname=file_)
    tar.close()
    
    s3.meta.client.upload_file(f'{args.output_dir}/model.tar.gz', bucket, prefix + 'model.tar.gz')
    
    tokenizer.save_pretrained(args.output_dir)
