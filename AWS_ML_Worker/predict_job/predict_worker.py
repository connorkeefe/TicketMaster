# pipeline/predict_worker.py

import os
import tempfile
from typing import Optional

import boto3
import botocore
import pandas as pd
from logger import logger


# You can also use environment variables if you want to change these without code edits
RAW_DAILY_PREFIX = os.getenv("RAW_DAILY_PREFIX", "raw_daily/")
FEATURE_STORE_PREFIX = os.getenv("FEATURE_STORE_PREFIX", "feature_store/TicketDailyFeatures/")
PREDICTIONS_PREFIX = os.getenv("PREDICTIONS_PREFIX", "predictions/")
METADATA_MODELS_PREFIX = os.getenv("METADATA_MODELS_PREFIX", "metadata/models/")


def _s3_key_startswith(key: str, prefix: str) -> bool:
    # normalize both just in case
    return key.startswith(prefix)


def _feature_store_key_for_raw(key: str) -> str:
    """
    Given a raw_daily key like:
        raw_daily/date=2025-11-29/raw_ticket_features.parquet

    Map it into the feature_store space:
        feature_store/TicketDailyFeatures/date=2025-11-29/raw_ticket_features.parquet
    """
    if not _s3_key_startswith(key, RAW_DAILY_PREFIX):
        # You can decide to error or just put under a default
        # For now, keep the full key but change prefix
        return f"{FEATURE_STORE_PREFIX}{key}"

    # strip raw_daily/ and prepend feature_store path
    suffix = key[len(RAW_DAILY_PREFIX):]
    return f"{FEATURE_STORE_PREFIX}{suffix.replace('raw_', '')}"


def _object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def _models_trained(s3_client, bucket: str, logger) -> bool:
    """
    Decide if we have *any* trained models yet.

    For now, we just check for the existence of the main LGBM 7d metadata file.
    You can extend this to require TFT + both LGBMs if you want stricter logic.
    """
    candidates = [
        f"{METADATA_MODELS_PREFIX}lgbm_7d_return_latest.json",
        f"{METADATA_MODELS_PREFIX}tft_section_7d_return_latest.json",
        f"{METADATA_MODELS_PREFIX}lgbm_14d_return_latest.json",
    ]

    found_any = False
    for key in candidates:
        if _object_exists(s3_client, bucket, key):
            logger.info("Detected trained model metadata at s3://%s/%s", bucket, key)
            found_any = True

    if not found_any:
        logger.info(
            "No model metadata found under s3://%s/%s; treating as 'no models trained yet'.",
            bucket,
            METADATA_MODELS_PREFIX,
        )

    return found_any


def _ingest_raw_into_feature_store(
    s3_client,
    bucket: str,
    raw_key: str,
    feature_store_key: str,
):
    """
    Ingest the new raw parquet into the feature store.

    First pass (simplest): just copy the parquet file 1:1 to the feature store path.
    Later you can evolve this to:
      - merge with existing partition
      - enforce schema
      - normalize timestamp, etc.
    """
    logger.info(
        "Ingesting raw data from s3://%s/%s into feature store s3://%s/%s",
        bucket,
        raw_key,
        bucket,
        feature_store_key,
    )
    s3_client.copy(
        {
            "Bucket": bucket,
            "Key": raw_key,
        },
        bucket,
        feature_store_key,
    )
    logger.info("Ingestion to feature store complete for s3://%s/%s", bucket, feature_store_key)


def _run_full_predict_flow(
    s3_client,
    bucket: str,
    raw_key: str,
    feature_store_key: str,
):
    """
    Placeholder for the *full* predict flow:

    - Download raw parquet
    - Append to feature_store (or rewrite partition)
    - Load latest TFT + LGBM models
    - Compute tft_section_signal for new rows
    - Run LGBM 7d/14d predictions
    - Write predictions parquet to predictions/...
    """
    logger.info(
        "Models detected. Running full predict flow for s3://%s/%s",
        bucket,
        raw_key,
    )

    # --- 1) Download raw parquet locally ---
    with tempfile.TemporaryDirectory() as tmpdir:
        local_raw = os.path.join(tmpdir, "raw.parquet")
        s3_client.download_file(bucket, raw_key, local_raw)
        df_new = pd.read_parquet(local_raw)

        logger.info("Loaded %d new rows from raw_daily parquet", len(df_new))

        # TODO: if needed, normalize/validate schema here.

        # --- 2) Append/overwrite into feature_store partition ---
        # For now, just upload the same file to feature_store.
        # Later you can merge with an existing partition using pandas, polars, or pyarrow.
        local_fs = os.path.join(tmpdir, "feature_store.parquet")
        df_new.to_parquet(local_fs, index=False)

        s3_client.upload_file(local_fs, bucket, feature_store_key)
        logger.info("Wrote new feature_store partition: s3://%s/%s", bucket, feature_store_key)

        # --- 3) TODO: Load TFT + LGBM models from metadata and run real predictions ---

        # Example stubs:
        # from .tft_inference import run_tft_on_new_rows
        # from .lgbm_inference import run_lgbm_predictions
        #
        # df_new = run_tft_on_new_rows(df_new, s3_client=s3_client, bucket=bucket, logger=logger)
        # df_new = run_lgbm_predictions(df_new, s3_client=s3_client, bucket=bucket, logger=logger)
        #
        # # --- 4) Save predictions snapshot ---
        # predictions_key = f"{PREDICTIONS_PREFIX}date={some_date}/job_id={some_job_id}.parquet"
        # df_new[["TicketDailyFeatureID", "tft_section_signal", "pred_7day_return", "pred_14day_return"]] \
        #     .to_parquet(local_preds, index=False)
        # s3_client.upload_file(local_preds, bucket, predictions_key)
        # logger.info("Uploaded predictions to s3://%s/%s", bucket, predictions_key)

        logger.info(
            "Full predict flow stub finished for s3://%s/%s (no actual ML run yet).",
            bucket,
            raw_key,
        )


def run_predict_pipeline(
    bucket: str,
    key: str,
    s3_client=None,
):
    """
    Entry point called from worker.process_predict_job.

    - Validates that key is under RAW_DAILY_PREFIX
    - Computes the corresponding feature_store key
    - If no models trained:
        -> only ingest into feature_store and return
    - If models exist:
        -> run full predict flow (currently stubbed)
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    logger.info("run_predict_pipeline started for s3://%s/%s", bucket, key)

    # Sanity check: only handle raw_daily keys here
    if not _s3_key_startswith(key, RAW_DAILY_PREFIX):
        logger.warning(
            "Key s3://%s/%s does not start with RAW_DAILY_PREFIX=%s. "
            "Skipping predict pipeline.",
            bucket,
            key,
            RAW_DAILY_PREFIX,
        )
        return True

    feature_store_key = _feature_store_key_for_raw(key)

    # Decide whether any models are actually trained yet
    if not _models_trained(s3_client, bucket, logger):
        # First stage of system: just load data into feature_store and exit
        _ingest_raw_into_feature_store(
            s3_client=s3_client,
            bucket=bucket,
            raw_key=key,
            feature_store_key=feature_store_key,
        )
        logger.info(
            "No models trained yet. Completed pure ingestion for s3://%s/%s",
            bucket,
            key,
        )
        return False

    # Later: execute full TFT + LGBM prediction flow
    _run_full_predict_flow(
        s3_client=s3_client,
        bucket=bucket,
        raw_key=key,
        feature_store_key=feature_store_key,
    )

    return False

    logger.info("run_predict_pipeline finished for s3://%s/%s", bucket, key)
