import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'sagemaker'])

from sagemaker.huggingface import HuggingFaceModel
from sagemaker import image_uris
import sagemaker
import argparse
import logging
import boto3
import json


class InferenceSpec:
    template = """
{    
    "InferenceSpecification": {
        "Containers" : [{"Image": "IMAGE_REPLACE_ME"}],
        "SupportedTransformInstanceTypes": INSTANCES_REPLACE_ME,
        "SupportedRealtimeInferenceInstanceTypes": INSTANCES_REPLACE_ME,
        "SupportedContentTypes": CONTENT_TYPES_REPLACE_ME,
        "SupportedResponseMIMETypes": RESPONSE_MIME_TYPES_REPLACE_ME
    }
}
"""

    def get_dict(self, ecr_image, supports_gpu, supported_content_types=None, supported_mime_types=None):
        return json.loads(self.get_json(ecr_image, 
                                        supports_gpu, 
                                        supported_content_types, 
                                        supported_mime_types))

    def get_json(self, ecr_image, supports_gpu, supported_content_types=None, supported_mime_types=None):
        if supported_mime_types is None:
            supported_mime_types = []
        if supported_content_types is None:
            supported_content_types = []
        return (self.template.replace("IMAGE_REPLACE_ME", ecr_image)
                             .replace("INSTANCES_REPLACE_ME", self.get_supported_instances(supports_gpu))
                             .replace("CONTENT_TYPES_REPLACE_ME", json.dumps(supported_content_types))
                             .replace("RESPONSE_MIME_TYPES_REPLACE_ME", json.dumps(supported_mime_types)))

    def get_supported_instances(self, supports_gpu):
        cpu_list = ["ml.m4.xlarge",
                    "ml.m4.2xlarge",
                    "ml.m4.4xlarge",
                    "ml.m4.10xlarge",
                    "ml.m4.16xlarge",
                    "ml.m5.large",
                    "ml.m5.xlarge",
                    "ml.m5.2xlarge",
                    "ml.m5.4xlarge",
                    "ml.m5.12xlarge",
                    "ml.m5.24xlarge",
                    "ml.c4.xlarge",
                    "ml.c4.2xlarge",
                    "ml.c4.4xlarge",
                    "ml.c4.8xlarge",
                    "ml.c5.xlarge",
                    "ml.c5.2xlarge",
                    "ml.c5.4xlarge",
                    "ml.c5.9xlarge",
                    "ml.c5.18xlarge"]
        gpu_list = [ "ml.p2.xlarge",
                     "ml.p2.8xlarge",
                     "ml.p2.16xlarge",
                     "ml.p3.2xlarge",
                     "ml.p3.8xlarge",
                     "ml.p3.16xlarge"]
        list_to_return = cpu_list
        if supports_gpu:
            list_to_return = cpu_list + gpu_list
        return json.dumps(list_to_return)


if __name__ == '__main__':
    # Parse argument variables passed via the DeployModel processing step
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str)
    parser.add_argument('--model_s3_path', type=str)
    parser.add_argument('--pipeline_name', type=str)
    parser.add_argument('--current_timestamp', type=str)
    args, _ = parser.parse_known_args()
    pipeline_name = args.pipeline_name
    current_timestamp = args.current_timestamp

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(level=logging.getLevelName('INFO'), 
                        handlers=[logging.StreamHandler(sys.stdout)], 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                       )
    
    region = args.region
    boto3.setup_default_session(region_name=region)
    role = sagemaker.get_execution_role()
    
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client('sagemaker')
    
    model_package_group_name = 'BERT-Email-Classifier'

    
    ecr_image = image_uris.retrieve(framework='huggingface', 
                                    region='us-east-1', 
                                    version='4.6.1', 
                                    image_scope='inference', 
                                    base_framework_version='tensorflow2.4.1', 
                                    py_version='py37', 
                                    container_version='ubuntu18.04', 
                                    instance_type='ml.m5.4xlarge')
    
    inference_spec = InferenceSpec().get_dict(ecr_image=ecr_image, 
                                              supports_gpu=False,  
                                              supported_content_types=['text/csv'], 
                                              supported_mime_types=['text/csv'])
    
    inference_spec["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] = 's3://sagemaker-us-east-1-892313895307/pipeline/model/model.tar.gz'
    
    model_metrics = {
    "ModelQuality": {
        "Statistics": {
            "ContentType": "application/json",
            "S3Uri": f"s3://sagemaker-us-east-1-892313895307/metrics/08-05-21-19/metrics.json",
        }
    }
}
    
    model_package_input_dict = {"ModelPackageGroupName": model_package_group_name, 
                                "ModelPackageDescription": f"CreatedOn: {current_timestamp}", 
                                "ModelApprovalStatus": "Approved", 
                                "ModelMetrics": model_metrics}

    model_package_input_dict.update(inference_spec)
    meta = {'MetadataProperties': {'GeneratedBy': 'NLP-Pipeline'}}

    model_package_input_dict.update(meta)
    
    sagemaker_client.create_model_package(**model_package_input_dict)