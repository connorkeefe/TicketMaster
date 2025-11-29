import sqlite3
from datetime import datetime, timedelta  # make sure this is imported at top of file
import pandas as pd
from run_flags import TEST
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger
from db.db_api import DB_API, DB_FILE


if TEST:
    DB_FILE_ML = '/Users/connorkeefe/PycharmProjects/TicketMaster/db/test_ml_database.db'
    DB_BACKUP_ML = '/Volumes/ConnorsDisk/TicketMasterDB/Backup/test/ML'
else:
    DB_FILE_ML = '/Users/connorkeefe/PycharmProjects/TicketMaster/db/ml_database.db'
    DB_BACKUP_ML = '/Volumes/ConnorsDisk/TicketMasterDB/Backup/ML'

CREATE_MODEL_TABLE = """
CREATE TABLE IF NOT EXISTS Model (
    ModelID TEXT PRIMARY KEY,
    Path TEXT,
    Type TEXT,
    TargetLabelName TEXT,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_TICKET_DAILY_FEATURES_TABLE = """
CREATE TABLE IF NOT EXISTS TicketDailyFeatures (
    TicketDailyFeatureID TEXT PRIMARY KEY,
    TicketID TEXT,
    EventID TEXT,
    TicketPriceRowID TEXT,
    SectionID TEXT,
    Timestamp TEXT,
    TimestampNorm TIMESTAMP,

    event_name TEXT,
    attraction_name_1 TEXT,
    attraction_name_2 TEXT,
    venue_name TEXT,
    segment TEXT,
    genre TEXT,
    subgenre TEXT,
    venue_city TEXT,
    venue_state TEXT,
    venue_country TEXT,
    event_day_of_week TEXT,

    Section TEXT,
    Row TEXT,
    Seat TEXT,

    sale_days_elapsed INTEGER,
    days_until_event INTEGER,
    day_of_week TEXT,
    attraction_1_rank INTEGER,
    attraction_2_rank INTEGER,

    ticket_price REAL,
    section_ticket_count INTEGER,
    section_min_price REAL,
    section_max_price REAL,
    section_avg_price REAL,
    section_median_price REAL,
    section_7day_median REAL,
    section_7day_price_vol REAL,
    section_7day_median_price_delta REAL,
    pct_below_section_median REAL,
    section_3days_sold INTEGER,
    section_3days_new INTEGER,

    tft_section_signal REAL,

    label_7day_section_median_return REAL,
    label_7day_section_median_price REAL,
    tft_train INTEGER DEFAULT 0,
    label_7day_price REAL,
    label_14day_price REAL,
    label_7day_return REAL,
    label_14day_return REAL,
    train_7_day INTEGER DEFAULT 0,
    train_14_day INTEGER DEFAULT 0,

    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_MODEL_PREDICTION_TABLE = """
CREATE TABLE IF NOT EXISTS ModelPrediction (
    ModelPredictionID TEXT PRIMARY KEY,
    Main_Model_ID TEXT REFERENCES Model(ModelID),
    Signal_Model_ID TEXT REFERENCES Model(ModelID),
    TicketDailyFeatureID TEXT REFERENCES TicketDailyFeatures(TicketDailyFeatureID),

    predict_7day_section_median_return REAL,
    predict_7day_section_median_price REAL,
    predict_7day_price REAL,
    predict_14day_price REAL,
    predict_7day_return REAL,
    predict_14day_return REAL,

    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_MODEL_ACCURACY_TABLE = """
CREATE TABLE IF NOT EXISTS ModelAccuracy (
    ModelAccuracyID TEXT PRIMARY KEY,
    ModelID TEXT REFERENCES Model(ModelID),
    number_predictions_sports INTEGER,
    number_predictions_concerts INTEGER,
    number_predictions_other INTEGER,
    eval_sports TEXT,
    eval_concerts TEXT,
    eval_other TEXT,
    total_predictions INTEGER,
    eval_total TEXT,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

class DB_API_ML:
    def __init__(self, db_file=None):
        """Initialize the database connection."""
        if db_file is None:
            db_file = DB_FILE_ML

        try:
            logger.info(f"Database connection establishing to {db_file}....")
            self.conn = sqlite3.connect(db_file, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            logger.info("Database connection established.")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            self.conn = None

    def create_tables(self):
        try:
            cur = self.conn.cursor()
            cur.execute(CREATE_MODEL_TABLE)
            cur.execute(CREATE_TICKET_DAILY_FEATURES_TABLE)
            cur.execute(CREATE_MODEL_ACCURACY_TABLE)
            cur.execute(CREATE_MODEL_PREDICTION_TABLE)
            self.conn.commit()
            logger.info(f"Created tables")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")

    def insert_ticket_daily_features_batch(self, df: pd.DataFrame):
        """
        Batch insert rows into TicketDailyFeatures.

        Assumes df has the columns produced by the feature query in build_ticket_daily_features.
        TicketDailyFeatureID is taken as TicketPriceRowID (1:1 with ticket price row).
        """
        if df.empty:
            logger.info("TicketDailyFeatures batch is empty, skipping insert.")
            return

        cur = self.conn.cursor()

        insert_sql = """
        INSERT OR IGNORE INTO TicketDailyFeatures (
            TicketDailyFeatureID,
            TicketID,
            EventID,
            TicketPriceRowID,
            SectionID,
            Timestamp,
            TimestampNorm,

            event_name,
            attraction_name_1,
            attraction_name_2,
            venue_name,
            segment,
            genre,
            subgenre,
            venue_city,
            venue_state,
            venue_country,
            event_day_of_week,

            Section,
            Row,
            Seat,

            sale_days_elapsed,
            days_until_event,
            day_of_week,
            attraction_1_rank,
            attraction_2_rank,

            ticket_price,
            section_ticket_count,
            section_min_price,
            section_max_price,
            section_avg_price,
            section_median_price,
            section_7day_median,
            section_7day_price_vol,
            section_7day_median_price_delta,
            pct_below_section_median,
            section_3days_sold,
            section_3days_new,
            tft_section_signal
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        );
        """

        params = []
        for _, r in df.iterrows():
            ts_raw = r["Timestamp"]
            ts_norm = build_sqlite_timestamp(ts_raw) if pd.notna(ts_raw) else None
            params.append(
                (
                    # IDs
                    r["TicketPriceRowID"],              # TicketDailyFeatureID
                    r["TicketID"],
                    r["EventID"],
                    r["TicketPriceRowID"],
                    r["SectionID"],
                    ts_raw,
                    ts_norm,

                    # Descriptive / categorical
                    r.get("event_name"),
                    r.get("attraction_name_1"),
                    r.get("attraction_name_2"),
                    r.get("venue_name"),
                    r.get("segment"),
                    r.get("genre"),
                    r.get("subgenre"),
                    r.get("venue_city"),
                    r.get("venue_state"),
                    r.get("venue_country"),
                    r.get("event_day_of_week"),

                    r.get("Section"),
                    r.get("Row"),
                    r.get("Seat"),

                    # Time-based features
                    int(r["sale_days_elapsed"]) if pd.notna(r["sale_days_elapsed"]) else None,
                    int(r["days_until_event"]) if pd.notna(r["days_until_event"]) else None,
                    r.get("day_of_week"),
                    int(r["attraction_1_rank"]) if pd.notna(r["attraction_1_rank"]) else None,
                    int(r["attraction_2_rank"]) if pd.notna(r["attraction_2_rank"]) else None,

                    # Price / section features
                    float(r["ticket_price"]) if pd.notna(r["ticket_price"]) else None,
                    int(r["section_ticket_count"]) if pd.notna(r["section_ticket_count"]) else None,
                    float(r["section_min_price"]) if pd.notna(r["section_min_price"]) else None,
                    float(r["section_max_price"]) if pd.notna(r["section_max_price"]) else None,
                    float(r["section_avg_price"]) if pd.notna(r["section_avg_price"]) else None,
                    float(r["section_median_price"]) if pd.notna(r["section_median_price"]) else None,
                    float(r["section_7day_median"]) if pd.notna(r["section_7day_median"]) else None,
                    float(r["section_7day_price_vol"]) if pd.notna(r["section_7day_price_vol"]) else None,
                    float(r["section_7day_median_price_delta"]) if pd.notna(r["section_7day_median_price_delta"]) else None,
                    float(r["pct_below_section_median"]) if pd.notna(r["pct_below_section_median"]) else None,
                    int(r["section_3days_sold"]) if pd.notna(r["section_3days_sold"]) else None,
                    int(r["section_3days_new"]) if pd.notna(r["section_3days_new"]) else None,
                    None  # tft_section_signal; to be filled after model inference
                )
            )

        cur.executemany(insert_sql, params)
        self.conn.commit()
        logger.info(f"Inserted {len(params)} TicketDailyFeatures rows (ignore on duplicates).")


    def drop_tables(self):
        try:
            cur = self.conn.cursor()
            cur.execute("PRAGMA foreign_keys = OFF;")
            # Get the list of all tables
            cur.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%';
            """)
            tables = cur.fetchall()

            # Drop each table
            for table in tables:
                table_name = table[0]
                cur.execute(f"DROP TABLE IF EXISTS {table_name};")
                logger.info(f"Table {table_name} dropped successfully.")

            self.conn.commit()
            logger.info("All tables deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")

    def backup_database(self) -> str | None:
        """
        Create a backup copy of the current SQLite database.

        Parameters:
            backup_dir (str): Directory where the backup file will be stored.
            include_timestamp (bool): Whether to append a timestamp to the backup filename.

        Returns:
            str | None: Path to the backup file if successful, otherwise None.
        """
        if not self.conn:
            logger.error("No active database connection. Backup aborted.")
            return None

        try:
            # Ensure backup directory exists
            os.makedirs(DB_BACKUP_ML, exist_ok=True)

            # Extract base DB filename
            base_name = os.path.basename(DB_FILE_ML).replace(".db", "")

            # Construct backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{base_name}_backup_{timestamp}.db"

            backup_path = os.path.join(DB_BACKUP_ML, backup_filename)

            # Perform safe SQLite backup
            with sqlite3.connect(backup_path) as backup_conn:
                self.conn.backup(backup_conn)

            logger.info(f"✅ Database backup created at: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"❌ Failed to back up database: {e}")
            return None


    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")



def build_sqlite_timestamp(ts_raw: str):
    """
    Convert 'YYYYMMDDHHMMSS-UUID' → 'YYYY-MM-DD HH:MM:SS' (SQLite-friendly),
    with a day-adjustment rule:

    - If the time is within 10 hours after midnight (00:00:00–09:59:59),
      treat it as belonging to the *previous* day.
    """
    if ts_raw is None:
        return None

    try:
        base = ts_raw[:14]  # 'YYYYMMDDHHMMSS'
        if len(base) < 14:
            return None

        year = int(base[0:4])
        month = int(base[4:6])
        day = int(base[6:8])
        hour = int(base[8:10])
        minute = int(base[10:12])
        second = int(base[12:14])

        dt = datetime(year, month, day, hour, minute, second)

        # If within 10 hours after "start of day", shift to previous day
        # i.e. 00:00:00–09:59:59 gets tagged as previous day
        if hour < 10:
            dt = dt - timedelta(days=1)

        return dt.strftime("%Y-%m-%d %H:%M:%S")

    except Exception:
        return None