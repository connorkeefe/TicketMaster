import os
from typing import Tuple
import json
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer, seed_everything         # <-- changed
from lightning.pytorch.callbacks import (                      # <-- changed
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_forecasting.data import NaNLabelEncoder
from AWS_ML_Worker.train_job.train_helpers import extract_date_from_key, append_metadata_key, read_parquet, write_parquet, read_metadata_keys, move_new_to_train_metadata, upload_checkpoint_to_s3, archive_existing_latest_metadata, write_latest_model_metadata
from logger import logger
import botocore
import io
from AWS_ML_Worker.constants import TFT_TRAIN_PREFIX, DAILY_FEATURE_PREFIX, TFT_NEW_DATA_META_KEY, TFT_TRAIN_DATA_META_KEY, BUCKET_NAME


# ---------- CONFIG ----------

TARGET_COL = "label_7day_section_median_return"   # adjust if your label column is named differently
GROUP_COL = "SectionID"
TIME_COL = "date"           # daily buckets
TIME_IDX_COL = "time_idx"   # integer time index for TFT
MODEL_PREFIX = "models/tft_section_7d_return/"
LATEST_META_KEY = "metadata/models/tft/latest_model.json"
ARCHIVE_META_PREFIX = "metadata/models/tft/archive/"

# MAX_ENCODER_LENGTH = 2     # days of history
# MIN_ENCODER_LENGTH = 1
# MAX_PREDICTION_LENGTH = 1   # we predict a 7-day-ahead return per day
# BATCH_SIZE = 128
# MAX_EPOCHS = 1
# # Lengths
MAX_ENCODER_LENGTH = 7      # 4-step lookback window
MIN_ENCODER_LENGTH = 3      # 1-step lookback window
MAX_PREDICTION_LENGTH = 1   # still one target per day (single-step)

# Training / hardware tuning for g4dn.xlarge
BATCH_SIZE = 300            # you can bump to 512 if GPU memory allows
MAX_EPOCHS = 30             # or whatever you want for “real” training





def build_tft_train_parquet_for_date(bucket: str, label_key: str, s3) -> str:
    """
    For a single label file (one date partition), build the TFT training parquet.

    Steps:
    - Parse date from label_key
    - Read labels parquet: must contain TicketDailyFeatureID + TARGET_COL
    - Read features parquet for that date: TicketDailyFeatures (TicketDailyFeatureID-level)
    - Join on TicketDailyFeatureID, then aggregate to SectionID+date
    - Save to tft/training/date=YYYY-MM-DD/tft_train.parquet

    Returns:
        The S3 key of the written training parquet.
    """
    date_str = extract_date_from_key(label_key)
    logger.info("Building TFT training parquet for date=%s", date_str)

    # 1) Load label parquet (TicketDailyFeatureID-level)
    df_labels = read_parquet(bucket, label_key, s3)

    if "TicketDailyFeatureID" not in df_labels.columns:
        raise ValueError(
            f"labels must contain TicketDailyFeatureID, got: {df_labels.columns.tolist()}"
        )
    if TARGET_COL not in df_labels.columns:
        raise ValueError(
            f"labels must contain '{TARGET_COL}', got: {df_labels.columns.tolist()}"
        )

    # 2) Load features parquet for same date (TicketDailyFeatures for that day)
    feature_key = f"{DAILY_FEATURE_PREFIX}date={date_str}/ticket_features.parquet"
    df_feat = read_parquet(bucket, feature_key, s3)

    if "TicketDailyFeatureID" not in df_feat.columns:
        raise ValueError(
            f"features must contain TicketDailyFeatureID, got: {df_feat.columns.tolist()}"
        )
    if GROUP_COL not in df_feat.columns:
        raise ValueError(
            f"features must contain SectionID, got: {df_feat.columns.tolist()}"
        )

    # Ensure 'date' column in features
    if TIME_COL not in df_feat.columns:
        if "TimestampNorm" in df_feat.columns:
            df_feat[TIME_COL] = pd.to_datetime(df_feat["TimestampNorm"]).dt.date.astype(str)
        else:
            df_feat[TIME_COL] = date_str

    if TARGET_COL in df_feat.columns:
        df_feat = df_feat.drop(columns=[TARGET_COL])
    # 3) Join labels into TicketDailyFeatures via TicketDailyFeatureID
    logger.info(f"LABELS COLUMNS: {df_labels.columns.tolist()}")

    df_join = df_feat.merge(
        df_labels,
        on="TicketDailyFeatureID",
        how="inner",
    )

    logger.info(f"JOIN COLUMNS: {df_join.columns.tolist()}")

    # 4) Aggregate ticket-level rows to section-level per day
    group_cols = [GROUP_COL, TIME_COL]

    agg_dict = {
        "event_name": "first",
        "attraction_name_1": "first",
        "attraction_name_2": "first",
        "venue_name": "first",
        "segment": "first",
        "genre": "first",
        "subgenre": "first",
        "venue_city": "first",
        "venue_state": "first",
        "venue_country": "first",
        "event_day_of_week": "first",

        "sale_days_elapsed": "first",
        "days_until_event": "first",
        "day_of_week": "first",
        "attraction_1_rank": "first",
        "attraction_2_rank": "first",

        "section_ticket_count": "first",
        "section_min_price": "first",
        "section_max_price": "first",
        "section_avg_price": "first",
        "section_median_price": "first",

        "section_7day_median": "first",
        "section_7day_price_vol": "first",
        "section_7day_median_price_delta": "first",
        "section_3days_sold": "first",
        "section_3days_new": "first",

        # label: identical for all TicketDailyFeatureID rows within a section/day bucket
        TARGET_COL: "first",
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in df_join.columns}

    df_section = (
        df_join
        .groupby(group_cols, as_index=False)
        .agg(agg_dict)
    )

    df_section = df_section[df_section[TARGET_COL].notna()].reset_index(drop=True)
    logger.info("Section-level rows for %s: %d", date_str, len(df_section))

    # 5) Save per-date TFT training parquet
    train_key = f"{TFT_TRAIN_PREFIX}date={date_str}/tft_train.parquet"
    write_parquet(df_section, bucket, train_key, s3)
    append_metadata_key(s3, bucket, TFT_NEW_DATA_META_KEY, train_key)

    return train_key

def debug_time_index(df: pd.DataFrame):
    # Just to be explicit
    print("TIME_IDX_COL:", TIME_IDX_COL, "GROUP_COL:", GROUP_COL)

    # Sort by group + time_idx
    df_sorted = df.sort_values([GROUP_COL, TIME_IDX_COL])

    # Look at a few groups
    some_groups = df_sorted[GROUP_COL].dropna().unique()[:5]
    for g in some_groups:
        tmp = df_sorted[df_sorted[GROUP_COL] == g][[GROUP_COL, TIME_COL, TIME_IDX_COL]].head(10)
        print(f"\n=== Group {g} (first 10 rows) ===")
        print(tmp)

    # Compute diffs per group
    df_sorted["idx_diff"] = df_sorted.groupby(GROUP_COL)[TIME_IDX_COL].diff()

    # Offenders: where diff > 1
    offenders = df_sorted[df_sorted["idx_diff"] > 1]
    print("\nNumber of offending rows (idx_diff > 1):", len(offenders))

    if not offenders.empty:
        print("First few offending rows:")
        print(
            offenders[[GROUP_COL, TIME_COL, TIME_IDX_COL, "idx_diff"]]
            .head(20)
            .to_string(index=False)
        )

        # Summary by group
        max_diff_by_group = (
            offenders.groupby(GROUP_COL)["idx_diff"]
            .max()
            .sort_values(ascending=False)
            .head(20)
        )
        print("\nMax idx_diff per offending group (top 20):")
        print(max_diff_by_group)



def build_tft_datasets(
    df: pd.DataFrame,
    val_cutoff_date: str | None = None,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Build PyTorch Forecasting TimeSeriesDataSet for train/validation.
    val_cutoff_date: if provided, use date < cutoff for train, >= cutoff for validation.
                     If None, will take last 10% of time_idx as validation.
    """
    # Ensure types
    df = df.copy()
    df[GROUP_COL] = df[GROUP_COL].astype(str)

    # Define which columns are what types
    static_categorical = ["event_name", "venue_name", "segment", "genre", "subgenre", "attraction_name_1",
            "attraction_name_2", "event_day_of_week", "sale_days_missing_flag"]

    time_varying_known_categoricals = [
        "day_of_week",  # of the price observation date
    ]
    time_varying_known_reals = [
        "sale_days_elapsed",  # days since sale start
        "days_until_event",  # days until event
    ]
    time_varying_unknown_reals = [
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
        TARGET_COL
    ]

    for col in df.columns:
        logger.info(
            "Column '%s': na_count=%d, na_frac=%.4f",
            col,
            df[col].isna().sum(),
            df[col].isna().mean(),
        )

    # 1. Flag: 0 if present, 1 if missing
    df["sale_days_missing_flag"] = df["sale_days_elapsed"].isna().astype(int)

    # 2. Replace missing sale_days_elapsed with 0
    df["sale_days_elapsed"] = df["sale_days_elapsed"].fillna(0)

    df = df.dropna(subset=["days_until_event"])
    df = df.dropna(subset=["event_name"])

    df["attraction_1_rank"] = df["attraction_1_rank"].fillna(200)
    df["attraction_2_rank"] = df["attraction_2_rank"].fillna(200)

    df["section_7day_price_vol"] = df["section_7day_price_vol"].fillna(0)
    df["section_3days_sold"] = df["section_3days_sold"].fillna(0)
    df["section_3days_new"] = df["section_3days_new"].fillna(0)


    # Add target to unknown reals (pytorch_forecasting best practice)
    if TARGET_COL not in time_varying_unknown_reals:
        time_varying_unknown_reals.append(TARGET_COL)

    for col in static_categorical:
        if col in df.columns:
            # print(col)
            df[col] = df[col].astype(str).fillna(f"__NA_{col}__")

    # train/val split
    if val_cutoff_date is not None:
        cutoff_ts = pd.to_datetime(val_cutoff_date)
        train_mask = df[TIME_COL] < cutoff_ts
        val_mask = ~train_mask
    else:
        # use last 10% of time_idx as validation
        cutoff_idx = df[TIME_IDX_COL].quantile(0.75)
        train_mask = df[TIME_IDX_COL] <= cutoff_idx
        val_mask = df[TIME_IDX_COL] > cutoff_idx

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    logger.info(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")
    # debug_time_index(train_df)

    training = TimeSeriesDataSet(
        train_df,
        time_idx=TIME_IDX_COL,
        target=TARGET_COL,
        group_ids=[GROUP_COL],
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_encoder_length=MIN_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=static_categorical,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=time_varying_unknown_reals,
        categorical_encoders={
            GROUP_COL: NaNLabelEncoder(add_nan=True),
            "event_name": NaNLabelEncoder(add_nan=True),
            "venue_name": NaNLabelEncoder(add_nan=True),
            "segment": NaNLabelEncoder(add_nan=True),
            "genre": NaNLabelEncoder(add_nan=True),
            "subgenre": NaNLabelEncoder(add_nan=True),
            "attraction_name_1": NaNLabelEncoder(add_nan=True),
            "attraction_name_2": NaNLabelEncoder(add_nan=True),
            "event_day_of_week": NaNLabelEncoder(add_nan=True),
            "sale_days_missing_flag": NaNLabelEncoder(add_nan=True),
            "day_of_week": NaNLabelEncoder(add_nan=True),
        },
        # normalize
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # --- robust validation construction ---
    try:
        validation = training.from_dataset(training, val_df)
    except AssertionError as e:
        logger.warning(
            "Failed to build validation dataset (all rows filtered out by encoder/decoder length etc.). "
            "Using training dataset as validation. Error: %s",
            e,
        )
        validation = training

    return training, validation


from datetime import datetime

def train_tft(
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    save_dir: str = "./tft_models",
) -> tuple[str, dict]:
    """
    Train a TemporalFusionTransformer on the given datasets.

    Returns:
        (best_checkpoint_path, eval_metrics_dict)
    """
    os.makedirs(save_dir, exist_ok=True)

    train_dataloader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=3
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=3
    )

    # IMPORTANT: construct TFT via from_dataset so it is a proper LightningModule
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=3e-4,
        hidden_size=48,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=24,
        loss=QuantileLoss(),
        optimizer="adam",
        log_interval=50,
        reduce_on_plateau_patience=4,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=5e-5,
        patience=10,
        verbose=True,
        mode="min",
    )

    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    seed_everything(42, workers=True)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",  # or "auto" once cuda is available
        devices=1,
        callbacks=[early_stop_callback, lr_logger, checkpoint_callback],
        gradient_clip_val=0.1,
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_path = checkpoint_callback.best_model_path
    logger.info(f"Best model saved to: {best_path}")

    # pull final val_loss from trainer callback_metrics
    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is not None:
        try:
            val_loss = float(val_loss.cpu().item())
        except Exception:
            val_loss = float(val_loss)
    else:
        val_loss = None

    eval_metrics = {
        "val_loss": val_loss,
        "best_checkpoint": os.path.basename(best_path),
        "epoch": int(trainer.current_epoch),
        "updated_at_utc": datetime.utcnow().isoformat() + "Z",
    }

    return best_path, eval_metrics

def register_trained_tft_model(
    s3,
    bucket: str,
    local_checkpoint_path: str,
    eval_metrics: dict,
) -> str:
    """
    - Uploads the checkpoint file to S3 under models/tft_section_7d_return/
    - Archives any existing latest_model.json
    - Writes new latest_model.json with metrics and model S3 key

    Returns:
        s3_model_key (where the checkpoint was uploaded)
    """
    # 1) Upload checkpoint
    s3_model_key = upload_checkpoint_to_s3(s3, bucket, local_checkpoint_path, MODEL_PREFIX)

    # 2) Archive previous latest_model.json if exists
    archive_existing_latest_metadata(s3, bucket, LATEST_META_KEY, ARCHIVE_META_PREFIX)

    # 3) Write new latest_model.json
    write_latest_model_metadata(s3, bucket, s3_model_key, eval_metrics, LATEST_META_KEY)

    return s3_model_key




def maybe_run_tft_training_from_metadata(
    s3,
    min_new_days: int = 1,
    save_dir: str = "/tmp/tft_models",
) -> str | None:
    """
    Check metadata/models/tft/new_data.txt:

    - If distinct dates in new_data < min_new_days → do nothing.
    - Else:
        * Build a new train set from train_data + new_data parquets.
        * Train TFT.
        * Move new_data keys into train_data metadata and clear new_data.

    Returns:
        best_model_path if training ran, otherwise None.
    """
    # 1) Read metadata
    new_keys   = read_metadata_keys(s3, BUCKET_NAME, TFT_NEW_DATA_META_KEY)
    train_keys = read_metadata_keys(s3, BUCKET_NAME, TFT_TRAIN_DATA_META_KEY)

    if not new_keys:
        logger.info("No new TFT training data (new_data.txt empty); skipping training.")
        return None

    # Distinct dates in the new data keys
    new_dates = sorted({extract_date_from_key(k) for k in new_keys})
    logger.info(
        "TFT new_data: %d keys across %d distinct dates: %s",
        len(new_keys), len(new_dates), new_dates
    )

    if len(new_dates) < min_new_days:
        logger.info(
            "Threshold not met: %d new days < min_new_days=%d. Not training this time.",
            len(new_dates), min_new_days
        )
        return None

    # 2) Decide which keys to train on this round
    #    Here: full retrain on all known data (existing train_keys + new_keys)
    all_keys = sorted(set(train_keys + new_keys))
    logger.info(
        "Training TFT on %d parquet files (%d existing, %d new).",
        len(all_keys), len(train_keys), len(new_keys)
    )

    # 3) Build full training dataframe from all_keys (pandas only)
    dfs = [read_parquet(BUCKET_NAME, key, s3) for key in all_keys]
    df_all = pd.concat(dfs, ignore_index=True)

    # Ensure time columns exist
    if TIME_COL not in df_all.columns:
        raise ValueError("Expected 'date' column in TFT train parquets.")

    df_all[TIME_COL]     = pd.to_datetime(df_all[TIME_COL])
    df_all[TIME_IDX_COL] = df_all[TIME_COL].dt.to_period("D").astype(int)

    # 4) Build TFT datasets and train
    training, validation = build_tft_datasets(df_all, val_cutoff_date=None)
    best_model_path, eval_metrics = train_tft(training, validation, save_dir=save_dir)
    logger.info("TFT training complete. Best model checkpoint: %s", best_model_path)

    # 5) Move new_data → train_data in metadata
    moved_new, updated_train = move_new_to_train_metadata(s3, BUCKET_NAME, TFT_NEW_DATA_META_KEY, TFT_TRAIN_DATA_META_KEY)
    logger.info(
        "Moved %d new keys into train_data; train_data now tracks %d keys.",
        len(moved_new), len(updated_train)
    )

    s3_model_key = register_trained_tft_model(
        s3=s3,
        bucket=BUCKET_NAME,
        local_checkpoint_path=best_model_path,
        eval_metrics=eval_metrics,
    )

    logger.info("Registered TFT model at s3://%s/%s", BUCKET_NAME, s3_model_key)

    return s3_model_key