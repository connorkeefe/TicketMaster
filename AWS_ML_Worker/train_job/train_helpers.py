import botocore
import pandas as pd
import json
import os
import io
from datetime import datetime
from logger import logger
import shutil
from botocore.exceptions import FlexibleChecksumError

def clear_tmp():
    tmp_path = "/tmp"
    for f in os.listdir(tmp_path):
        full = os.path.join(tmp_path, f)
        try:
            if os.path.isfile(full) or os.path.islink(full):
                os.remove(full)
            elif os.path.isdir(full):
                shutil.rmtree(full)
        except Exception as e:
            logger.info(f"Could not remove {full}: {e}")


def read_metadata_keys(s3, bucket: str, meta_key: str) -> list[str]:
    """
    Read newline-delimited S3 keys from metadata file.
    If file doesn't exist, return [].
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=meta_key)
        body = obj["Body"].read().decode("utf-8")
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        return lines
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return []
        raise

def append_metadata_key(s3, bucket: str, meta_key: str, new_key: str) -> None:
    """
    Append an S3 key to a newline-delimited metadata file, avoiding duplicates.
    """
    keys = read_metadata_keys(s3, bucket, meta_key)
    if new_key not in keys:
        keys.append(new_key)
    body = "".join(k + "\n" for k in keys)
    s3.put_object(Bucket=bucket, Key=meta_key, Body=body.encode("utf-8"))

def write_metadata_keys(s3, bucket: str, meta_key: str, keys: list[str]) -> None:
    body = "".join(k + "\n" for k in keys)
    s3.put_object(Bucket=bucket, Key=meta_key, Body=body.encode("utf-8"))

def extract_date_from_key(key: str) -> str:
    """
    Extract YYYY-MM-DD from a key like:
      labels/section_return_7d/date=2025-11-27/labels.parquet
    """
    parts = key.split("/")
    for p in parts:
        if p.startswith("date="):
            return p.split("=", 1)[1]  # '2025-11-27'
    raise ValueError(f"Could not find date= in key: {key}")



def read_parquet(bucket: str, key: str, s3) -> pd.DataFrame:
    logger.info("Reading parquet from s3://%s/%s", bucket, key)

    buf = io.BytesIO()
    # this is the same code path that works in save_mock_s3_to_local
    s3.download_fileobj(bucket, key, buf)

    # rewind to the start
    buf.seek(0)

    df = pd.read_parquet(buf)
    logger.info("Loaded parquet with %d rows, %d cols", len(df), len(df.columns))
    return df


def write_parquet(df: pd.DataFrame, bucket: str, key: str, s3) -> None:
    logger.info("Writing parquet to s3://%s/%s (rows=%d)", bucket, key, len(df))
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

def move_new_to_train_metadata(s3, bucket: str, new_key, train_key) -> tuple[list[str], list[str]]:
    """
    Move all keys from new_data.txt into train_data.txt.
    Returns (new_keys, updated_train_keys).
    """
    new_keys   = read_metadata_keys(s3, bucket, new_key)
    train_keys = read_metadata_keys(s3, bucket, train_key)

    if not new_keys:
        return [], train_keys

    updated_train = sorted(set(train_keys + new_keys))

    write_metadata_keys(s3, bucket, train_key, updated_train)
    write_metadata_keys(s3, bucket, new_key, [])  # clear

    return new_keys, updated_train

def read_json_if_exists(s3, bucket: str, key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise

def upload_checkpoint_to_s3(s3, bucket: str, local_path: str, model_prefix: str) -> str:
    """
    Upload the local checkpoint file to S3 under MODEL_PREFIX.
    Returns the S3 key.
    """
    filename = os.path.basename(local_path)
    s3_key = f"{model_prefix}{filename}"

    logger.info("Uploading model checkpoint %s to s3://%s/%s", local_path, bucket, s3_key)
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=s3_key, Body=f)

    return s3_key

def archive_existing_latest_metadata(s3, bucket: str, meta_key: str, meta_archive: str) -> None:
    """
    If latest_model.json exists, move its contents to
    metadata/models/tft/archive/{timestamp}_model.json
    and remove latest_model.json.
    """
    latest = read_json_if_exists(s3, bucket, meta_key)
    if latest is None:
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_key = f"{meta_archive}{ts}_model.json"

    logger.info(
        "Archiving existing latest_model.json to s3://%s/%s",
        bucket,
        archive_key,
    )

    s3.put_object(
        Bucket=bucket,
        Key=archive_key,
        Body=json.dumps(latest, indent=2).encode("utf-8"),
    )
    s3.delete_object(Bucket=bucket, Key=meta_key)

def write_latest_model_metadata(
    s3,
    bucket: str,
    s3_model_key: str,
    eval_metrics: dict,
    meta_key: str,
) -> None:
    """
    Overwrite metadata/models/tft/latest_model.json with
    info about the newly trained model.
    """
    payload = {
        "model_s3_key": s3_model_key,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "metrics": eval_metrics,
    }

    logger.info(
        "Writing latest model metadata to s3://%s/%s",
        bucket,
        meta_key,
    )
    s3.put_object(
        Bucket=bucket,
        Key=meta_key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
    )

