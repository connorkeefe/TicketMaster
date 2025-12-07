import os

def get_env(name: str, default=None, required: bool = False):
    value = os.getenv(name, default)
    if required and value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

LOCAL_TEST_MODE = os.getenv("LOCAL_TEST_MODE", "false").lower() in ("1", "true", "yes")

# When running locally, we'll synthesize SQS messages from S3 objects
BUCKET_NAME = get_env("BUCKET_NAME", "ticketmaster-ml-pipeline")
MOCK_PREDICT_PREFIX = get_env("MOCK_PREDICT_PREFIX", "raw_daily/")
MOCK_TRAIN_PREFIX = get_env("MOCK_TRAIN_PREFIX", "labels/")


PREDICT_QUEUE_URL = get_env("PREDICT_QUEUE_URL", required=False, default=None)
TRAIN_QUEUE_URL   = get_env("TRAIN_QUEUE_URL", required=False, default=None)
SNS_TOPIC_ARN     = get_env("SNS_TOPIC_ARN", required=False, default=None)
ASG_NAME          = get_env("ASG_NAME", required=False, default=None)

IDLE_TIMEOUT_SECONDS      = int(get_env("IDLE_TIMEOUT_SECONDS", "50"))
TRAIN_BUFFER_SECONDS      = int(get_env("IDLE_TIMEOUT_SECONDS", "30"))
VISIBILITY_TIMEOUT_SECONDS = int(get_env("VISIBILITY_TIMEOUT_SECONDS", "3600"))
MAX_MESSAGES_PER_POLL     = 1

DAILY_FEATURE_PREFIX = get_env("DAILY_FEATURE_PREFIX", "feature_store/TicketDailyFeatures/")

TFT_TRAIN_PREFIX   = get_env("TFT_TRAIN_PREFIX",   "tft/training/")
TFT_NEW_DATA_META_KEY   = "metadata/models/tft/new_data.txt"
TFT_TRAIN_DATA_META_KEY = "metadata/models/tft/train_data.txt"
