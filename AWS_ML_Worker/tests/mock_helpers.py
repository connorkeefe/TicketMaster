import os

import boto3
import json
from logger import logger

from AWS_ML_Worker.constants import MOCK_PREDICT_PREFIX, MOCK_TRAIN_PREFIX, BUCKET_NAME, MAX_MESSAGES_PER_POLL

def load_local_folder_into_mock_s3(s3):
    local_root = "/Users/connorkeefe/PycharmProjects/TicketMaster/AWS_ML_Worker/tests/data/s3"
    bucket_name = BUCKET_NAME
    s3.create_bucket(Bucket=bucket_name)

    for root, _, files in os.walk(os.path.join(local_root, bucket_name)):
        for filename in files:
            if filename == ".DS_Store":
                continue
            full_path = os.path.join(root, filename)
            key = os.path.relpath(full_path, os.path.join(local_root, bucket_name))
            key = key.replace(os.sep, "/")  # Windows-safe
            local_size = os.path.getsize(full_path)
            # logger.info("Local file %s (%d bytes) -> s3://%s/%s",
            #             full_path, local_size, bucket_name, key)

            s3.upload_file(full_path, bucket_name, key)
            # sanity-check what’s in mock S3
            head = s3.head_object(Bucket=bucket_name, Key=key)
            s3_size = head["ContentLength"]
            # logger.info("S3 object size for %s: %d bytes", key, s3_size)
            logger.info(f"📁 uploaded -> s3://{bucket_name}/{key}")
            if s3_size == 0 and local_size > 0:
                logger.error("‼ Uploaded 0-byte object but local file is non-empty!")

    logger.info(f"🎉 Finished loading local folder into mock S3 bucket: {bucket_name}")

def save_mock_s3_to_local(s3):
    local_root = "/Users/connorkeefe/PycharmProjects/TicketMaster/AWS_ML_Worker/tests/data/s3"
    bucket_name = BUCKET_NAME

    resp = s3.list_objects_v2(Bucket=bucket_name)
    contents = resp.get("Contents", []) or []

    for obj in contents:
        key = obj["Key"]
        local_path = os.path.join(local_root, bucket_name, key)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, "wb") as f:
            s3.download_fileobj(bucket_name, key, f)

        logger.info(f"⬇️ saved -> {local_path}")

    logger.info(f"💾 Finished writing mock S3 state back to folder {local_root}")

def build_mock_messages(queue_name: str, s3) -> list[dict]:
    """
    Local test helper:
    - For 'predict': list objects under MOCK_PREDICT_PREFIX
    - For 'train':   list objects under MOCK_TRAIN_PREFIX
    - Assumes:
        * LOCAL_TEST_MODE is True
        * Exactly one bucket exists in the (mocked) S3
    - Builds fake SQS messages whose Body mimics an S3 event JSON.
    """
    # We assume moto is already started and a single bucket exists
    bucket = s3.list_buckets()["Buckets"][0]["Name"]

    assert bucket == BUCKET_NAME

    if queue_name == "predict":
        prefix = MOCK_PREDICT_PREFIX
    elif queue_name == "train":
        prefix = MOCK_TRAIN_PREFIX
    else:
        # super simple: nothing fancy
        logger.warning("Unknown queue_name=%s in build_mock_messages", queue_name)
        return []

    logger.info("Building mock %s messages from s3://%s/%s", queue_name, bucket, prefix)

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    # logger.info(resp)
    contents = resp.get("Contents", []) or []

    messages: list[dict] = []
    for i, obj in enumerate(contents[:]):
        key = obj["Key"]
        body = json.dumps(
            {
                "Records": [
                    {
                        "s3": {
                            "bucket": {"name": bucket},
                            "object": {"key": key},
                        }
                    }
                ]
            }
        )
        messages.append(
            {
                "Body": body,
                "ReceiptHandle": f"mock-{queue_name}-{i}",
            }
        )

    logger.info("Created %d mock %s messages.", len(messages), queue_name)
    return messages