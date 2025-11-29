"""
worker.py

Long-running worker process for the TicketML pipeline.

Responsibilities:
- Poll Predict + Train SQS queues
- For each S3 event message:
    - Determine job type
    - Call the appropriate pipeline function
- Delete messages only on success
- Send SNS notifications
- When both queues are idle for a while, scale ASG DesiredCapacity=0 and exit
"""

import json
import logging
import os
import time
import traceback
from typing import Iterable, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from urllib.parse import unquote_plus


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_env(name: str, default=None, required: bool = False):
    value = os.getenv(name, default)
    if required and value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


PREDICT_QUEUE_URL = get_env("PREDICT_QUEUE_URL", required=True)
TRAIN_QUEUE_URL = get_env("TRAIN_QUEUE_URL", required=True)
SNS_TOPIC_ARN = get_env("SNS_TOPIC_ARN", default=None)
ASG_NAME = get_env("ASG_NAME", required=True)

IDLE_TIMEOUT_SECONDS = int(get_env("IDLE_TIMEOUT_SECONDS", "50"))          # 15 min
VISIBILITY_TIMEOUT_SECONDS = int(get_env("VISIBILITY_TIMEOUT_SECONDS", "3600"))
MAX_MESSAGES_PER_POLL = 1    # keep it simple for now


# ---------------------------------------------------------------------------
# AWS clients
# ---------------------------------------------------------------------------

session = boto3.Session()
sqs = session.client("sqs")
sns = session.client("sns")
autoscaling = session.client("autoscaling")
s3 = session.client("s3")   # you may or may not use it directly here


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ticketml.worker")


# ---------------------------------------------------------------------------
# S3 event parsing utilities
# ---------------------------------------------------------------------------

def iter_s3_records(message_body: str) -> Iterable[Tuple[str, str]]:
    """
    Given an SQS message body (string JSON from S3 event),
    yield (bucket, key) tuples for each record.
    """
    try:
        payload = json.loads(message_body)
    except json.JSONDecodeError:
        logger.error("Message body is not valid JSON, skipping: %s", message_body[:500])
        return []

    records = payload.get("Records", [])
    for rec in records:
        try:
            bucket = rec["s3"]["bucket"]["name"]
            key = unquote_plus(rec["s3"]["object"]["key"])
            yield bucket, key
        except KeyError:
            logger.warning("Malformed S3 record in message: %s", rec)


def classify_label_type(s3_key: str) -> str | None:
    """
    Decide which label type a key corresponds to based on its prefix.
    You can adjust this to your exact S3 layout.
    """
    if s3_key.startswith("labels/return_7day/") or s3_key.startswith("labels/return_7d/"):
        return "return_7d"
    if s3_key.startswith("labels/return_14day/") or s3_key.startswith("labels/return_14d/"):
        return "return_14d"
    if s3_key.startswith("labels/section_return_7d/"):
        return "section_return_7d"
    return None


# ---------------------------------------------------------------------------
# Pipeline hooks (YOU implement these functions in your own modules)
# ---------------------------------------------------------------------------

def process_predict_job(bucket: str, key: str) -> None:
    """
    Here you call your prediction pipeline.

    Typical steps:
    - Load new raw_daily parquet from s3://bucket/key
    - Build/append TicketDailyFeatures rows
    - Run TFT to compute tft_section_signal
    - Run LGBM 7d/14d models using that feature
    - Write predictions back to S3
    - Optionally write predictions back to your SQLite / Postgres DB
    """
    logger.info("Processing PREDICT job for s3://%s/%s", bucket, key)

    # TODO: replace with real implementation
    # from pipeline.predict_worker import run_predict_pipeline
    # run_predict_pipeline(bucket, key, s3_client=s3)
    time.sleep(2)  # placeholder so you can see something in logs
    logger.info("Predict job completed for s3://%s/%s", bucket, key)


def process_train_job(bucket: str, key: str) -> None:
    """
    Here you call your training pipeline.

    Steps:
    - Load label file (TicketDailyFeatureID → label value) from s3://bucket/key
    - Deduplicate / merge into feature store
    - For each label type present:
        - Update training pool
        - If enough new rows, retrain TFT / LGBM model
        - Save new model + metadata + train/test snapshots
    """
    label_type = classify_label_type(key)
    logger.info(
        "Processing TRAIN job for s3://%s/%s (label_type=%s)", bucket, key, label_type
    )

    # TODO: replace with real implementation
    # from pipeline.train_worker import run_train_pipeline
    # run_train_pipeline(bucket, key, label_type, s3_client=s3)
    time.sleep(2)
    logger.info("Train job completed for s3://%s/%s", bucket, key)


# ---------------------------------------------------------------------------
# SNS helper
# ---------------------------------------------------------------------------

def send_notification(subject: str, message: str) -> None:
    if not SNS_TOPIC_ARN:
        logger.info("SNS_TOPIC_ARN not set; skipping notification: %s", subject)
        return

    try:
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject[:100],  # SNS subject limit
            Message=message,
        )
    except (BotoCoreError, ClientError) as e:
        logger.error("Failed to send SNS notification: %s", e)


# ---------------------------------------------------------------------------
# SQS polling + job routing
# ---------------------------------------------------------------------------

def poll_queue(queue_url: str) -> list[dict]:
    """
    Long-poll the given SQS queue and return a list of messages (may be empty).
    """
    try:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=MAX_MESSAGES_PER_POLL,
            WaitTimeSeconds=20,                 # long polling
            VisibilityTimeout=VISIBILITY_TIMEOUT_SECONDS,
        )
    except (BotoCoreError, ClientError) as e:
        logger.error("Error receiving messages from SQS (%s): %s", queue_url, e)
        return []

    return resp.get("Messages", [])


def handle_message(queue_name: str, msg: dict) -> bool:
    """
    Route a single SQS message to appropriate handler.
    Returns True if processed successfully, False otherwise.
    """
    body = msg.get("Body", "")
    receipt_handle = msg["ReceiptHandle"]

    # S3 put → SQS sends S3 Event JSON directly in Body.
    # If you later wrap via SNS, you may need to unwrap one more level.
    success = True

    try:
        for bucket, key in iter_s3_records(body):
            if queue_name == "predict":
                process_predict_job(bucket, key)
            elif queue_name == "train":
                process_train_job(bucket, key)
            else:
                logger.error("Unknown queue_name=%s", queue_name)
                success = False
    except Exception:
        logger.error(
            "Exception while processing %s message:\n%s",
            queue_name,
            traceback.format_exc(),
        )
        success = False

    if success:
        try:
            sqs.delete_message(
                QueueUrl=PREDICT_QUEUE_URL if queue_name == "predict" else TRAIN_QUEUE_URL,
                ReceiptHandle=receipt_handle,
            )
            logger.info("Deleted %s message from queue", queue_name)
        except (BotoCoreError, ClientError) as e:
            logger.error("Failed to delete %s message: %s", queue_name, e)
            # message will reappear after visibility timeout

    return success


# ---------------------------------------------------------------------------
# ASG scale-down logic
# ---------------------------------------------------------------------------

def scale_down_asg():
    """
    Set the ASG desired capacity to 0 so this instance gets terminated.
    """
    logger.info("Idle timeout reached. Scaling ASG %s down to 0.", ASG_NAME)
    try:
        autoscaling.update_auto_scaling_group(
            AutoScalingGroupName=ASG_NAME,
            DesiredCapacity=0,
        )
        send_notification(
            subject="TicketML worker scaled down",
            message=f"Worker reached idle timeout. ASG {ASG_NAME} set to DesiredCapacity=0.",
        )
    except (BotoCoreError, ClientError) as e:
        logger.error("Failed to scale down ASG: %s", e)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    logger.info("TicketML worker starting up.")
    logger.info("Predict queue: %s", PREDICT_QUEUE_URL)
    logger.info("Train queue:   %s", TRAIN_QUEUE_URL)
    logger.info("ASG name:      %s", ASG_NAME)
    logger.info("Idle timeout:  %s seconds", IDLE_TIMEOUT_SECONDS)

    last_work_time = time.time()

    while True:
        did_work = False

        # 1) Poll predict queue first (so new raw data gets predictions quickly)
        predict_messages = poll_queue(PREDICT_QUEUE_URL)
        if predict_messages:
            did_work = True
            for msg in predict_messages:
                handle_message("predict", msg)

        # 2) Then poll train queue
        train_messages = poll_queue(TRAIN_QUEUE_URL)
        if train_messages:
            did_work = True
            for msg in train_messages:
                handle_message("train", msg)

        if did_work:
            last_work_time = time.time()
            continue

        # No work this cycle
        idle_for = time.time() - last_work_time
        if idle_for >= IDLE_TIMEOUT_SECONDS:
            logger.info("No messages for %.1f seconds. Exiting worker.", idle_for)
            scale_down_asg()
            break

        # Sleep a bit before polling again
        time.sleep(10)


if __name__ == "__main__":
    main()
