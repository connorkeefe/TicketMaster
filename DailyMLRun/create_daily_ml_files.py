import os
import sys
import argparse
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import boto3

from db.db_api_ml import DB_API_ML  # uses DB_FILE_ML from your config
from logger import logger
AWS_UPLOAD = True

BATCH = False

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

RAW_DAILY_PREFIX = "raw_daily/"
LABELS_PREFIX = "labels/"

LOCAL_S3_ROOT = (
    "/Users/connorkeefe/PycharmProjects/TicketMaster/AWS_ML_Worker/tests/data/s3"
)
BUCKET_NAME = "ticketmaster-ml-pipeline"


LABEL_SPECS = [
    # (label_column_name, subfolder)
    ("label_7day_return", "return_7d", 7),
    ("label_14day_return", "return_14d", 14),
    ("label_7day_section_median_return", "section_return_7d", 7),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def get_s3_client():
    """Create a boto3 S3 client (real AWS)."""
    session = boto3.Session()
    return session.client("s3", region_name="us-east-1")


def write_parquet(
    df: pd.DataFrame,
    key: str,
    s3_client=None,
):
    """
    Write a DataFrame either to real S3 (aws_upload=True) or to a local
    folder that mirrors S3 layout (aws_upload=False).

    key: e.g. 'raw_daily/date=YYYY-MM-DD/raw_ticket_features.parquet'
    """
    if df.empty:
        logger.info("DataFrame is empty for key=%s, skipping write.", key)
        return

    if AWS_UPLOAD:
        if s3_client is None:
            s3_client = get_s3_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "tmp.parquet")
            df.to_parquet(local_path, index=False)
            logger.info("Uploading to s3://%s/%s", BUCKET_NAME, key)
            s3_client.upload_file(local_path, BUCKET_NAME, key)
    else:
        # Local test: write to folder such that:
        #   local_s3_root/bucket_name/<key>
        dest_path = os.path.join(LOCAL_S3_ROOT, BUCKET_NAME, *key.split("/"))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        logger.info("Writing local parquet: %s", dest_path)
        df.to_parquet(dest_path, index=False)


def export_for_date(
    conn,
    date_str: str,
    s3_client=None,
):
    """
    Export both PREDICT and TRAIN parquets for a single date, based on
    date(TimestampNorm) = date_str.

    - Predict parquet (all columns):
        raw_daily/date=YYYY-MM-DD/raw_ticket_features.parquet
    - Label mapping parquets:
        labels/return_7d/date=YYYY-MM-DD/labels.parquet
        labels/return_14d/date=YYYY-MM-DD/labels.parquet
        labels/section_return_7d/date=YYYY-MM-DD/labels.parquet
    """
    logger.info("Exporting data for date(TimestampNorm) = %s", date_str)

    # --------------------- PREDICT PARQUET ----------------------------
    df_predict = pd.read_sql_query(
        """
        SELECT *
        FROM TicketDailyFeatures
        WHERE date(TimestampNorm) = ?
        """,
        conn,
        params=(date_str,),
    )

    logger.info(
        "Predict export: %d rows from TicketDailyFeatures for date %s",
        len(df_predict),
        date_str,
    )

    predict_key = f"{RAW_DAILY_PREFIX}date={date_str}/raw_ticket_features.parquet"
    write_parquet(
        df_predict,
        key=predict_key,
        s3_client=s3_client,
    )

    # --------------------- TRAIN PARQUETS (labels) --------------------
    for label_col, subfolder, back_days in LABEL_SPECS:
        label_date = datetime.strptime(date_str, "%Y-%m-%d").date() - timedelta(days=back_days)
        label_date_str = label_date.strftime("%Y-%m-%d")
        df_label = pd.read_sql_query(
            f"""
            SELECT TicketDailyFeatureID, {label_col}
            FROM TicketDailyFeatures
            WHERE date(TimestampNorm) = ?
              AND {label_col} IS NOT NULL
            """,
            conn,
            params=(label_date_str,),
        )

        logger.info(
            "Label export [%s]: %d rows for date %s back_days=%s, label_date=%s",
            label_col,
            len(df_label),
            date_str,
            back_days,
            label_date_str
        )

        if df_label.empty:
            continue

        label_key = (
            f"{LABELS_PREFIX}{subfolder}/date={label_date_str}/labels.parquet"
        )
        write_parquet(
            df_label,
            key=label_key,
            s3_client=s3_client,
        )


def get_all_distinct_dates(conn):
    """
    Get all distinct dates from TimestampNorm as 'YYYY-MM-DD' strings.
    """
    df_dates = pd.read_sql_query(
        """
        SELECT DISTINCT date(TimestampNorm) AS d
        FROM TicketDailyFeatures
        WHERE TimestampNorm IS NOT NULL
        ORDER BY d DESC;
        """,
        conn,
    )
    # drop any NULLs just in case
    df_dates = df_dates.dropna(subset=["d"])
    return df_dates["d"].tolist()


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------


def main():
    logger.info("Batch mode is: %s", BATCH)
    logger.info("AWS upload: %s", AWS_UPLOAD)
    logger.info("Bucket: %s", BUCKET_NAME)
    if not AWS_UPLOAD:
        logger.info("Local S3 root: %s", LOCAL_S3_ROOT)

    s3_client = get_s3_client() if AWS_UPLOAD else None

    db = DB_API_ML()  # picks DB_FILE_ML internally
    conn = db.conn

    try:
        if BATCH:
            # Export for ALL dates we have
            dates = get_all_distinct_dates(conn)
            logger.info("Found %d distinct date(TimestampNorm) values.", len(dates))
            for d in dates:
                export_for_date(
                    conn=conn,
                    date_str=d,
                    s3_client=s3_client,
                )


        else:
            yesterday = datetime.utcnow().date() - timedelta(days=1)
            date_str = yesterday.strftime("%Y-%m-%d")

            logger.info("Incremental export date: %s", date_str)

            export_for_date(
                conn=conn,
                date_str=date_str,
                s3_client=s3_client,
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()
