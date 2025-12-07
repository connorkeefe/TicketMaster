# AWS_ML_Worker/train_job/offline_full_tft_train.py

import os
import boto3

from logger import logger
from AWS_ML_Worker.constants import BUCKET_NAME, LOCAL_TEST_MODE
from AWS_ML_Worker.train_job.tft_training import (
    build_tft_train_parquet_for_date,
    maybe_run_tft_training_from_metadata,
)
from AWS_ML_Worker.train_job.train_helpers import write_metadata_keys, clear_tmp
from moto import mock_aws
from tests.mock_helpers import load_local_folder_into_mock_s3, save_mock_s3_to_local

# 🔹 NEW: import predict pipeline + RAW prefix
from predict_job.predict_worker import run_predict_pipeline, RAW_DAILY_PREFIX

s3 = None
session = None
mock = None

if LOCAL_TEST_MODE:
    mock = mock_aws()
    mock.start()

# create a session in both modes
session = boto3.Session(
    region_name=os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)

# shared s3 client
s3 = session.client("s3")

# only real AWS clients when not local
# sqs = session.client("sqs") if not LOCAL_TEST_MODE else None
# sns = session.client("sns") if not LOCAL_TEST_MODE else None
# autoscaling = session.client("autoscaling") if not LOCAL_TEST_MODE else None

# in LOCAL_TEST_MODE, create the bucket and load local files using this SAME s3 client
if LOCAL_TEST_MODE:
    load_local_folder_into_mock_s3(s3)

# Labels are now the driver
LABEL_PREFIX = "labels/section_return_7d/"


def list_label_keys(s3, bucket: str, prefix: str) -> list[str]:
    """
    List all label parquet files under the given prefix.
    Typically these will look like:
      labels/section_return_7d/date=YYYY-MM-DD/labels.parquet
    """
    keys: list[str] = []
    continuation_token = None

    while True:
        params = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        resp = s3.list_objects_v2(**params)
        contents = resp.get("Contents", []) or []

        for obj in contents:
            key = obj["Key"]
            # tighten filter if you only want specific filenames
            if key.endswith(".parquet"):
                keys.append(key)

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    keys = sorted(keys)
    logger.info("Found %d label parquet(s) under %s", len(keys), prefix)
    return keys


# 🔹 NEW: list_raw_keys helper to drive predict pipeline from raw_daily
def list_raw_keys(s3, bucket: str, prefix: str) -> list[str]:
    """
    List all raw_daily parquet files under the given prefix.
    Typically these will look like:
      raw_daily/date=YYYY-MM-DD/raw_ticket_features.parquet
    """
    keys: list[str] = []
    continuation_token = None

    while True:
        params = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        resp = s3.list_objects_v2(**params)
        contents = resp.get("Contents", []) or []

        for obj in contents:
            key = obj["Key"]
            if key.endswith(".parquet"):
                keys.append(key)

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    keys = sorted(keys)
    logger.info("Found %d raw_daily parquet(s) under %s", len(keys), prefix)
    return keys


def main():
    logger.info("==== Offline full TFT training starting (label-driven) ====")

    # (Optional) If you want a fully fresh run, uncomment:
    # from AWS_ML_Worker.constants import TFT_NEW_DATA_META_KEY, TFT_TRAIN_DATA_META_KEY
    # write_metadata_keys(s3, BUCKET_NAME, TFT_NEW_DATA_META_KEY, [])
    # write_metadata_keys(s3, BUCKET_NAME, TFT_TRAIN_DATA_META_KEY, [])

    # 🔹 0) First, run predict pipeline on all raw_daily files
    #     so that feature_dfs / feature_store partitions are created.
    raw_keys = list_raw_keys(s3, BUCKET_NAME, RAW_DAILY_PREFIX)

    if not raw_keys:
        logger.warning(
            "No raw_daily parquet files found under s3://%s/%s; skipping predict pipeline.",
            BUCKET_NAME,
            RAW_DAILY_PREFIX,
        )
    else:
        logger.info("Running predict pipeline for %d raw_daily file(s)...", len(raw_keys))
        for raw_key in raw_keys:
            try:
                run_predict_pipeline(bucket=BUCKET_NAME, key=raw_key, s3_client=s3)
            except Exception as e:
                logger.exception(
                    "Predict pipeline failed for raw key %s: %s",
                    raw_key,
                    e,
                )

    # 1) Discover all label files
    label_keys = list_label_keys(s3, BUCKET_NAME, LABEL_PREFIX)

    if not label_keys:
        logger.warning(
            "No label parquet files found under s3://%s/%s; nothing to do for TFT training.",
            BUCKET_NAME,
            LABEL_PREFIX,
        )
        return

    # 2) For each label file, build per-date TFT train parquet
    for label_key in label_keys:
        try:
            train_key = build_tft_train_parquet_for_date(
                bucket=BUCKET_NAME,
                label_key=label_key,
                s3=s3,
            )
            if train_key is None:
                logger.info(
                    "No non-NaN labels in %s – skipped building TFT train parquet.",
                    label_key,
                )
            else:
                logger.info(
                    "Built TFT train parquet %s from labels %s",
                    train_key,
                    label_key,
                )
        except Exception as e:
            logger.exception(
                "Failed to build TFT train parquet for labels %s: %s",
                label_key,
                e,
            )

    # 3) Run TFT training over all train_data + new_data
    try:
        model_s3_key = maybe_run_tft_training_from_metadata(
            s3=s3,
            min_new_days=1,          # treat any amount of data as enough for training
            save_dir="/tmp/tft_models",
        )
        if model_s3_key is None:
            logger.info("No TFT training run (threshold not met or no new data).")
        else:
            logger.info(
                "TFT training complete. Model registered at s3://%s/%s",
                BUCKET_NAME,
                model_s3_key,
            )
    except Exception as e:
        logger.error("TFT training failed: %s", e)
        # optionally clear /tmp here if you want
        # clear_tmp()

    save_mock_s3_to_local(s3)

    logger.info("==== Offline full TFT training finished ====")


if __name__ == "__main__":
    main()
