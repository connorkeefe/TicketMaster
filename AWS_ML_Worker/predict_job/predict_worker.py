# pipeline/predict_worker.py
import os
import tempfile
from typing import Optional
import re
import boto3
import botocore
import pandas as pd

from AWS_ML_Worker.train_job.tft_training import MAX_ENCODER_LENGTH, MIN_ENCODER_LENGTH
from logger import logger
from datetime import datetime, timedelta
import io
import json
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data.encoders import TorchNormalizer  # NEW
import torch
from torch.serialization import add_safe_globals  # NEW


MAX_ENCODER_LENGTH = 7
MIN_ENCODER_LENGTH = 3
# You can also use environment variables if you want to change these without code edits
RAW_DAILY_PREFIX = os.getenv("RAW_DAILY_PREFIX", "raw_daily/")
FEATURE_STORE_PREFIX = os.getenv("FEATURE_STORE_PREFIX", "feature_store/TicketDailyFeatures/")
PREDICTIONS_PREFIX = os.getenv("PREDICTIONS_PREFIX", "predictions/")
METADATA_MODELS_PREFIX = os.getenv("METADATA_MODELS_PREFIX", "metadata/models/")

TFT_LATEST_MODEL_META_KEY = os.getenv(
    "TFT_LATEST_MODEL_META_KEY",
    "metadata/models/tft/latest_model.json",
)
# Column names must match your TFT training setup
TFT_TARGET_COL = "label_7day_section_median_return"
TFT_GROUP_COL = "SectionID"
TFT_TIME_COL = "date"          # daily buckets
TFT_TIME_IDX_COL = "time_idx"  # integer time index (days)

# Where TFT checkpoints live (written by train pipeline)
TFT_MODEL_PREFIX = os.getenv("TFT_MODEL_PREFIX", "models/tft_section_7d_return/")

# Where label parquets live for per-day accuracy (same structure as training)
TFT_LABEL_PREFIX = os.getenv("TFT_LABEL_PREFIX", "labels/section_return_7d/")

# Where to store the rolling CSV of daily accuracy / prediction counts
TFT_DAILY_METRICS_KEY = os.getenv(
    "TFT_DAILY_METRICS_KEY",
    "metadata/models/tft/daily_scores.csv",
)

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
        # f"{METADATA_MODELS_PREFIX}tft/latest_model.json",
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

_TFT_MODEL_CACHE = None
_TFT_MODEL_S3_KEY = None


def _get_tft_checkpoint_key_from_latest_metadata(
    s3_client,
    bucket: str,
) -> Optional[str]:
    """
    Read metadata/tft/latest_model.json and return the `model_s3_key` field.

    JSON structure example:
    {
      "model_s3_key": "models/tft_section_7d_return/tft-epoch=00-val_loss=0.0359.ckpt",
      "created_at_utc": "2025-12-07T17:03:35.394470Z",
      "metrics": { ... }
    }
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=TFT_LATEST_MODEL_META_KEY)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.warning(
                "No latest TFT model metadata found at s3://%s/%s",
                bucket,
                TFT_LATEST_MODEL_META_KEY,
            )
            return None
        raise

    body = obj["Body"].read().decode("utf-8")
    try:
        meta = json.loads(body)
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse JSON in s3://%s/%s: %s",
            bucket,
            TFT_LATEST_MODEL_META_KEY,
            e,
        )
        return None

    model_key = meta.get("model_s3_key")
    if not model_key:
        logger.error(
            "latest_model.json missing 'model_s3_key' field at s3://%s/%s",
            bucket,
            TFT_LATEST_MODEL_META_KEY,
        )
        return None

    logger.info(
        "Using TFT checkpoint from latest_model.json: s3://%s/%s",
        bucket,
        model_key,
    )
    return model_key



def _load_latest_tft_model(s3_client, bucket: str):
    """
    Download and load the TFT checkpoint specified in metadata/tft/latest_model.json.
    Cached in memory so we only download once per process.
    """
    global _TFT_MODEL_CACHE, _TFT_MODEL_S3_KEY

    # Return cached instance if we already loaded it
    if _TFT_MODEL_CACHE is not None:
        return _TFT_MODEL_CACHE

    ckpt_key = _get_tft_checkpoint_key_from_latest_metadata(s3_client, bucket)
    if ckpt_key is None:
        logger.warning("No TFT checkpoint key available from latest_model.json")
        return None

    local_ckpt = os.path.join(tempfile.gettempdir(), os.path.basename(ckpt_key))

    # Download checkpoint if not already present
    if not os.path.exists(local_ckpt):
        logger.info("Downloading TFT checkpoint s3://%s/%s to %s", bucket, ckpt_key, local_ckpt)
        s3_client.download_file(bucket, ckpt_key, local_ckpt)
    else:
        logger.info("Reusing local TFT checkpoint file at %s", local_ckpt)

    # 🔐 Register TorchNormalizer as a safe global for torch.load(weights_only=True)
    try:
        add_safe_globals([TorchNormalizer])
        logger.info("Registered TorchNormalizer as a safe global for torch.load.")
    except Exception as e:
        logger.warning("Failed to register TorchNormalizer as safe global: %s", e)

    map_location = "cuda" if torch.cuda.is_available() else "cpu"

    # ⚠️ We explicitly set weights_only=False to restore the pre-2.6 behavior.
    # This bypasses the new safe-globals restrictions and will load the full checkpoint.
    # Only do this because you trust your own checkpoint.
    tft = TemporalFusionTransformer.load_from_checkpoint(
        local_ckpt,
        map_location=map_location,
        weights_only=False,
    )

    _TFT_MODEL_CACHE = tft
    _TFT_MODEL_S3_KEY = ckpt_key

    logger.info(
        "Loaded TFT model from s3://%s/%s (map_location=%s)",
        bucket,
        ckpt_key,
        map_location,
    )
    return tft



# ----------------------
# METRICS CSV UPDATER
# ----------------------

def _append_tft_daily_metrics_row(
    s3_client,
    bucket: str,
    record: dict,
):
    """
    Append a single row to the TFT daily metrics CSV in S3.
    Columns: date, has_labels, n_predictions, n_labeled, mae, mse, created_at_utc
    """
    # Ensure created_at_utc
    record = dict(record)
    record["created_at_utc"] = datetime.utcnow().isoformat() + "Z"

    # Load existing CSV if present
    existing_df: Optional[pd.DataFrame] = None
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=TFT_DAILY_METRICS_KEY)
        body = obj["Body"].read()
        existing_df = pd.read_csv(io.BytesIO(body))
        logger.info(
            "Loaded existing TFT daily metrics CSV (%d rows) from s3://%s/%s",
            len(existing_df),
            bucket,
            TFT_DAILY_METRICS_KEY,
        )
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.info(
                "No existing TFT daily metrics CSV at s3://%s/%s – creating new.",
                bucket,
                TFT_DAILY_METRICS_KEY,
            )
            existing_df = None
        else:
            raise

    new_row_df = pd.DataFrame([record])
    if existing_df is not None:
        out_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        out_df = new_row_df

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpf:
        out_df.to_csv(tmpf.name, index=False)
        tmpf.flush()
        s3_client.upload_file(tmpf.name, bucket, TFT_DAILY_METRICS_KEY)

    logger.info(
        "Updated TFT daily metrics CSV with row for date=%s at s3://%s/%s",
        record.get("date"),
        bucket,
        TFT_DAILY_METRICS_KEY,
    )

# ----------------------
# PER-PARTITION TFT PREDICTION + METRICS
# ----------------------

def _extract_date_from_feature_store_key(feature_store_key: str) -> Optional[str]:
    """
    Extract YYYY-MM-DD from keys like:
      feature_store/TicketDailyFeatures/date=2025-11-29/ticket_features.parquet
    """
    m = re.search(r"date=(\d{4}-\d{2}-\d{2})", feature_store_key)
    return m.group(1) if m else None


def _run_tft_on_feature_store_partition(
    s3_client,
    bucket: str,
    feature_store_key: str,
):
    """
    For a given feature_store partition (one date):

    1) Load a multi-day window of feature_store data around this date
       (last TFT_MAX_ENCODER_LENGTH + TFT_MAX_PREDICTION_LENGTH days).
    2) Aggregate to SectionID+date (section-level) for ALL those days.
    3) Run TFT.predict() on the multi-day frame.
    4) Keep only predictions for this specific date.
    5) Merge those section-level predictions back into the *ticket-level*
       feature_store parquet for this date as `tft_section_signal`.
    6) If labels exist, compute MAE/MSE and append to the daily metrics CSV.
    """

    tft = _load_latest_tft_model(s3_client, bucket)
    if tft is None:
        logger.warning(
            "TFT model not available; skipping TFT prediction for %s",
            feature_store_key,
        )
        return

    date_str = _extract_date_from_feature_store_key(feature_store_key)
    if date_str is None:
        logger.warning(
            "Could not parse date from feature_store key %s – skipping TFT prediction.",
            feature_store_key,
        )
        return

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    logger.info(
        "Running TFT prediction for feature_store partition s3://%s/%s (date=%s)",
        bucket,
        feature_store_key,
        date_str,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # ------------------------------------------------------------
        # 1) Load multi-day feature_store history around target date
        # ------------------------------------------------------------
        day_dfs: list[pd.DataFrame] = []
        num_days_loaded = 0

        total_window = MAX_ENCODER_LENGTH + 1
        for offset in range(total_window):
            d = target_date - timedelta(days=offset)
            d_str = d.isoformat()

            hist_key = (
                f"{FEATURE_STORE_PREFIX}date={d_str}/"
                "ticket_features.parquet"
            )

            if not _object_exists(s3_client, bucket, hist_key):
                continue

            local_hist = os.path.join(tmpdir, f"fs_{d_str}.parquet")
            s3_client.download_file(bucket, hist_key, local_hist)
            df_d = pd.read_parquet(local_hist)

            # Ensure date column exists for this partition
            if TFT_TIME_COL not in df_d.columns:
                if "TimestampNorm" in df_d.columns:
                    df_d[TFT_TIME_COL] = (
                        pd.to_datetime(df_d["TimestampNorm"])
                        .dt.date.astype(str)
                    )
                else:
                    df_d[TFT_TIME_COL] = d_str

            day_dfs.append(df_d)
            num_days_loaded += 1

        if not day_dfs:
            logger.warning(
                "No feature_store history found for date=%s; skipping TFT prediction.",
                date_str,
            )
            # Record a metrics row with zero predictions
            record = {
                "date": date_str,
                "has_labels": False,
                "n_predictions": 0,
                "n_labeled": 0,
                "mae": None,
                "mse": None,
            }
            _append_tft_daily_metrics_row(s3_client, bucket, record)
            return

        df_hist = pd.concat(day_dfs, ignore_index=True)
        logger.info(
            "Loaded %d rows from %d day(s) of feature_store history for TFT.",
            len(df_hist),
            num_days_loaded,
        )

        # ------------------------------------------------------------
        # 2) Aggregate to SectionID+date (section-level) for ALL days
        # ------------------------------------------------------------
        required_cols = [
            "event_name",
            "attraction_name_1",
            "attraction_name_2",
            "venue_name",
            "segment",
            "genre",
            "subgenre",
            "venue_city",
            "venue_state",
            "venue_country",
            "event_day_of_week",
            "sale_days_elapsed",
            "days_until_event",
            "day_of_week",
            "attraction_1_rank",
            "attraction_2_rank",
            "section_ticket_count",
            "section_min_price",
            "section_max_price",
            "section_avg_price",
            "section_median_price",
            "section_7day_median",
            "section_7day_price_vol",
            "section_7day_median_price_delta",
            "section_3days_sold",
            "section_3days_new",
        ]

        group_cols = [TFT_GROUP_COL, TFT_TIME_COL]
        agg_dict = {c: "first" for c in required_cols if c in df_hist.columns}

        df_section_all = (
            df_hist.groupby(group_cols, as_index=False)
                   .agg(agg_dict)
        )
        logger.info(
            "Constructed section-level TFT input: %d rows (SectionID+date) over %d day(s).",
            len(df_section_all),
            num_days_loaded,
        )

        # ------------------------------------------------------------
        # 3) Mirror training-time cleaning logic
        # ------------------------------------------------------------

        # sale_days_missing_flag
        if "sale_days_elapsed" in df_section_all.columns:
            df_section_all["sale_days_missing_flag"] = (
                df_section_all["sale_days_elapsed"].isna().astype(int)
            )
            df_section_all["sale_days_elapsed"] = (
                df_section_all["sale_days_elapsed"].fillna(0)
            )

        if "days_until_event" in df_section_all.columns:
            df_section_all = df_section_all.dropna(subset=["days_until_event"])

        if "event_name" in df_section_all.columns:
            df_section_all = df_section_all.dropna(subset=["event_name"])

        if "attraction_1_rank" in df_section_all.columns:
            df_section_all["attraction_1_rank"] = (
                df_section_all["attraction_1_rank"].fillna(200)
            )
        if "attraction_2_rank" in df_section_all.columns:
            df_section_all["attraction_2_rank"] = (
                df_section_all["attraction_2_rank"].fillna(200)
            )

        for col in ["section_7day_price_vol", "section_3days_sold", "section_3days_new"]:
            if col in df_section_all.columns:
                df_section_all[col] = df_section_all[col].fillna(0)

        # Timestamps / time_idx
        df_section_all[TFT_TIME_COL] = pd.to_datetime(df_section_all[TFT_TIME_COL])
        df_section_all[TFT_TIME_IDX_COL] = (
            df_section_all[TFT_TIME_COL].dt.to_period("D").astype(int)
        )

        # Ensure group ID is str
        df_section_all[TFT_GROUP_COL] = df_section_all[TFT_GROUP_COL].astype(str)

        # Ensure target column exists and is numeric (can be all NaN)
        if TFT_TARGET_COL not in df_section_all.columns:
            df_section_all[TFT_TARGET_COL] = pd.NA

        df_section_all[TFT_TARGET_COL] = pd.to_numeric(
            df_section_all[TFT_TARGET_COL],
            errors="coerce",
        )

        # Static categoricals as strings
        for col in [
            "event_name",
            "venue_name",
            "segment",
            "genre",
            "subgenre",
            "attraction_name_1",
            "attraction_name_2",
            "event_day_of_week",
            "sale_days_missing_flag",
        ]:
            if col in df_section_all.columns:
                df_section_all[col] = (
                    df_section_all[col].astype(str).fillna(f"__NA_{col}__")
                )

        # ------------------------------------------------------------
        # 4) Run TFT.predict on the multi-day section-level frame
        # ------------------------------------------------------------
        logger.info(
            "Calling TFT.predict on %d section-level rows (multi-day window)...",
            len(df_section_all),
        )

        try:
            preds = tft.predict(df_section_all, mode="prediction")
        except AssertionError as e:
            # This is where you previously hit:
            # "filters should not remove entries all entries..."
            logger.warning(
                "TFT.predict filtered out all series for date=%s. "
                "Likely not enough history yet. Skipping TFT. Error: %s",
                date_str,
                e,
            )
            record = {
                "date": date_str,
                "has_labels": False,
                "n_predictions": 0,
                "n_labeled": 0,
                "mae": None,
                "mse": None,
            }
            _append_tft_daily_metrics_row(s3_client, bucket, record)
            return

        if hasattr(preds, "numpy"):
            preds = preds.numpy()
        preds = pd.Series(preds.reshape(-1), index=df_section_all.index, name="tft_section_signal")
        df_section_all["tft_section_signal"] = preds

        logger.info(
            "TFT predictions complete for %d rows in the multi-day frame.",
            len(df_section_all),
        )

        # ------------------------------------------------------------
        # 5) Keep only predictions for the TARGET DATE
        # ------------------------------------------------------------
        target_ts = pd.to_datetime(date_str)

        df_section_all[TFT_TIME_COL] = pd.to_datetime(df_section_all[TFT_TIME_COL])
        df_section_today = df_section_all.loc[
            df_section_all[TFT_TIME_COL] == target_ts,
            [TFT_GROUP_COL, TFT_TIME_COL, "tft_section_signal"],
        ].copy()

        n_pred = df_section_today["tft_section_signal"].notna().sum()
        logger.info(
            "Filtered to %d section-level predictions for target date %s.",
            n_pred,
            date_str,
        )

        # ------------------------------------------------------------
        # 6) Merge section-level predictions into ticket-level partition
        # ------------------------------------------------------------
        local_fs_today = os.path.join(tmpdir, "feature_store_today.parquet")
        s3_client.download_file(bucket, feature_store_key, local_fs_today)
        df_today = pd.read_parquet(local_fs_today)

        # Ensure date + group columns on ticket-level df
        if TFT_TIME_COL not in df_today.columns:
            if "TimestampNorm" in df_today.columns:
                df_today[TFT_TIME_COL] = (
                    pd.to_datetime(df_today["TimestampNorm"])
                    .dt.date.astype(str)
                )
            else:
                df_today[TFT_TIME_COL] = date_str

        df_today[TFT_TIME_COL] = pd.to_datetime(df_today[TFT_TIME_COL])
        df_today[TFT_GROUP_COL] = df_today[TFT_GROUP_COL].astype(str)

        df_today = df_today.merge(
            df_section_today,
            on=[TFT_GROUP_COL, TFT_TIME_COL],
            how="left",
        )

        n_pred_ticket = df_today["tft_section_signal"].notna().sum()
        logger.info(
            "Merged TFT predictions into ticket-level df: %d rows with tft_section_signal.",
            n_pred_ticket,
        )

        # Overwrite feature_store partition with new column
        df_today.to_parquet(local_fs_today, index=False)
        s3_client.upload_file(local_fs_today, bucket, feature_store_key)
        logger.info(
            "Updated feature_store partition with TFT signal at s3://%s/%s",
            bucket,
            feature_store_key,
        )

        # ------------------------------------------------------------
        # 7) Accuracy metrics if labels exist for this date
        # ------------------------------------------------------------
        label_key = f"{TFT_LABEL_PREFIX}date={date_str}/labels.parquet"
        has_labels = False
        n_labeled = 0
        mae = None
        mse = None

        try:
            if _object_exists(s3_client, bucket, label_key):
                logger.info(
                    "Found labels for date=%s at s3://%s/%s – computing accuracy metrics.",
                    date_str,
                    bucket,
                    label_key,
                )
                local_labels = os.path.join(tmpdir, "labels.parquet")
                s3_client.download_file(bucket, label_key, local_labels)
                df_labels = pd.read_parquet(local_labels)

                if (
                    "TicketDailyFeatureID" in df_today.columns
                    and "TicketDailyFeatureID" in df_labels.columns
                ):
                    merged = df_today.merge(
                        df_labels[["TicketDailyFeatureID", TFT_TARGET_COL]],
                        on="TicketDailyFeatureID",
                        how="inner",
                    )
                else:
                    logger.warning(
                        "TicketDailyFeatureID not present in df_today or labels; "
                        "skipping accuracy computation for date=%s.",
                        date_str,
                    )
                    merged = None

                if merged is not None:
                    mask = (
                        merged["tft_section_signal"].notna()
                        & merged[TFT_TARGET_COL].notna()
                    )
                    merged = merged[mask]
                    n_labeled = len(merged)
                    if n_labeled > 0:
                        diff = merged["tft_section_signal"] - merged[TFT_TARGET_COL]
                        mae = float(diff.abs().mean())
                        mse = float((diff ** 2).mean())
                        has_labels = True
                        logger.info(
                            "Date=%s: TFT metrics on %d labeled rows – MAE=%.6f, MSE=%.6f",
                            date_str,
                            n_labeled,
                            mae,
                            mse,
                        )
            else:
                logger.info(
                    "No label parquet for date=%s at s3://%s/%s – recording prediction count only.",
                    date_str,
                    bucket,
                    label_key,
                )
        except Exception as e:
            logger.exception(
                "Error while computing TFT accuracy metrics for date=%s: %s",
                date_str,
                e,
            )

    # ------------------------------------------------------------
    # 8) Record metrics / counts in CSV
    # ------------------------------------------------------------
    record = {
        "date": date_str,
        "has_labels": bool(has_labels),
        "n_predictions": int(n_pred_ticket),
        "n_labeled": int(n_labeled),
        "mae": mae,
        "mse": mse,
    }
    _append_tft_daily_metrics_row(s3_client, bucket, record)


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
        local_fs = os.path.join(tmpdir, "feature_store.parquet")
        df_new.to_parquet(local_fs, index=False)

        s3_client.upload_file(local_fs, bucket, feature_store_key)
        logger.info("Wrote new feature_store partition: s3://%s/%s", bucket, feature_store_key)

    # --- NEW: run TFT on this feature_store partition + log metrics ---
    _run_tft_on_feature_store_partition(
        s3_client=s3_client,
        bucket=bucket,
        feature_store_key=feature_store_key,
    )

    logger.info(
        "Full predict flow finished for s3://%s/%s (TFT prediction + metrics logged).",
        bucket,
        raw_key,
    )




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
