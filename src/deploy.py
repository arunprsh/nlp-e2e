from sagemaker.huggingface import HuggingFaceModel
import sagemaker
import argparse
import boto3
import time


role = sagemaker.get_execution_role()

# Parse argument variables passed via the DeployModel processing step
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--region', type=str)
parser.add_argument('--deployment_instance_type', type=str)
parser.add_argument('--deployment_instance_count', type=int)
parser.add_argument('--model_s3_path', type=str)
parser.add_argument('--endpoint_name', type=str)
args = parser.parse_args()

region = args.region


# Create HuggingFaceModel
huggingface_model = HuggingFaceModel(model_data=f'{args.model_S3_path}/model.tar.gz',  # path to your trained sagemaker model 
                                     role=role, # iam role with permissions to create an Endpoint 
                                     transformers_version='4.6', # transformers version used 
                                     pytorch_version='1.7', # pytorch version used 
                                     py_version='py36', # python version of the DLC
                                    )

huggingface_model.deploy(initial_instance_count=args.deployment_instance_count, 
                         instance_type=args.deployment_instance_type)