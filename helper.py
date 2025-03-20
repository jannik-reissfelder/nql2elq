import boto3
import json
from botocore.exceptions import ClientError
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
import pandas as pd
import os


def get_elastic_secret():
    secret_name = "SECRET_KEY_ELASTIC"
    region_name = "eu-central-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = get_secret_value_response['SecretString']
    elastic_password = json.loads(secret)["ELASTIC_PASSWORD"]
    elastic_cloud_id = json.loads(secret)["CLOUD_ID"]

    return elastic_password, elastic_cloud_id



def init_elastic_client():
    # Get Elasticsearch credentials
    elastic_password, elastic_cloud_id = get_elastic_secret()

    client = Elasticsearch(
        cloud_id=elastic_cloud_id,
        basic_auth=("elastic", elastic_password),
        timeout=60,
        max_retries=10,  # Set the maximum number of retries
        retry_on_timeout=True
    )
    return client



def get_openai_secret(secret_key):
    secret_name = secret_key
    region_name = "eu-central-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = json.loads(get_secret_value_response['SecretString'])
    return secret

