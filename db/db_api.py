import sqlite3
from datetime import datetime, timedelta

import pandas
import pandas as pd
from run_flags import TEST
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger

# Database file path - you can modify this path as needed
# DB_FILE = '/Volumes/ConnorsDisk/TicketMasterDB/Working/events_database.db'
if TEST:
    DB_FILE = '/Users/connorkeefe/PycharmProjects/TicketMaster/db/test_events_database.db'
    DB_BACKUP = '/Volumes/ConnorsDisk/TicketMasterDB/Backup/test/Event'
else:
    DB_FILE = '/Users/connorkeefe/PycharmProjects/TicketMaster/db/events_database.db'
    DB_BACKUP = '/Volumes/ConnorsDisk/TicketMasterDB/Backup/Event'

class Median:
    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            self.values.append(value)

    def finalize(self):
        if not self.values:
            return None
        self.values.sort()
        n = len(self.values)
        mid = n // 2
        if n % 2 == 1:
            return float(self.values[mid])
        else:
            return (self.values[mid - 1] + self.values[mid]) / 2.0

def build_sqlite_timestamp(ts_raw: str):
    """
    Convert 'YYYYMMDDHHMMSS-UUID' → 'YYYY-MM-DD HH:MM:SS'
    """
    if ts_raw is None:
        return None

    base = ts_raw[:14]  # first 14 chars = YYYYMMDDHHMMSS

    # Build formatted string
    try:
        return (
            base[0:4] + "-" +
            base[4:6] + "-" +
            base[6:8] + " " +
            base[8:10] + ":" +
            base[10:12] + ":" +
            base[12:14]
        )
    except:
        return None


# SQL statements to create tables (SQLite compatible)
CREATE_VENUES_TABLE = """
CREATE TABLE IF NOT EXISTS Venues (
    VenueID TEXT PRIMARY KEY,
    name TEXT,
    city TEXT,
    state TEXT,
    country TEXT
);
"""

CREATE_ATTRACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS Attractions (
    AttractionID TEXT PRIMARY KEY,
    name TEXT
);
"""

CREATE_ATTRACTION_INSTANCE_TABLE = """
CREATE TABLE IF NOT EXISTS AttractionInstanceDetail (
    AttractionInstanceID TEXT PRIMARY KEY,
    AttractionID1 TEXT REFERENCES Attractions(AttractionID),
    Attraction1Rank INTEGER,
    AttractionID2 TEXT REFERENCES Attractions(AttractionID),
    Attraction2Rank INTEGER
);
"""

CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS Events (
    EventID TEXT PRIMARY KEY,
    VenueID TEXT REFERENCES Venues(VenueID),
    URL TEXT,
    AttractionID1 TEXT REFERENCES Attractions(AttractionID),
    AttractionID2 TEXT REFERENCES Attractions(AttractionID),
    sale_start TIMESTAMP,
    sale_end TIMESTAMP,
    name TEXT,
    event_start TIMESTAMP,
    timezone TEXT,
    segment TEXT,
    genre TEXT,
    subgenre TEXT,
    type TEXT,
    subtype TEXT
);
"""

CREATE_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS Prices (
    Timestamp TEXT PRIMARY KEY,
    EventID TEXT REFERENCES Events(EventID),
    AttractionInstanceID TEXT REFERENCES AttractionInstanceDetail(AttractionInstanceID),
    Min_Price REAL,
    Max_Price REAL,
    Currency TEXT,
    Stan_Min_Price REAL,
    Stan_Max_Price REAL,
    Stan_Count INTEGER,
    Resl_Min_Price REAL,
    Resl_Max_Price REAL,
    Resl_Count INTEGER
);
"""

CREATE_PRICE_RECORDS_TABLE = """
CREATE TABLE IF NOT EXISTS PriceRecords (
    Timestamp TEXT,
    EventID TEXT,
    AttractionInstanceID TEXT,
    PRIMARY KEY (Timestamp, EventID, AttractionInstanceID)
);
"""

CREATE_TICKETS_TABLE = """
CREATE TABLE IF NOT EXISTS Tickets (
    EventID TEXT REFERENCES Events(EventID),
    Section TEXT,
    Row TEXT,
    Seat TEXT,
    TicketID TEXT PRIMARY KEY,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_TICKET_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS TicketPrices (
    TicketPriceRowID TEXT PRIMARY KEY,
    Timestamp TEXT REFERENCES Prices(Timestamp),
    TicketID TEXT REFERENCES Tickets(TicketID),
    TicketPrice REAL,
    Currency TEXT,
    SellableQuantity TEXT,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_SECTION_TABLE = """
CREATE TABLE IF NOT EXISTS Section (
    SectionID TEXT PRIMARY KEY,
    EventID TEXT REFERENCES Events(EventID),
    Section TEXT,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_SECTION_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS SectionPrices (
    SectionPriceRowID TEXT PRIMARY KEY,
    Timestamp TEXT REFERENCES Prices(Timestamp),
    SectionID TEXT REFERENCES Section(SectionID),
    MedianPrice REAL,
    AveragePrice REAL,
    Currency TEXT,
    NumberOfTickets INTEGER,
    MaxPrice REAL,
    MinPrice REAL,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_SECTION_MARKET_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS SectionMarketData (
    SectionMarketDataID TEXT PRIMARY KEY,
    SectionPriceRowID TEXT REFERENCES SectionPrices(SectionPriceRowID),
    sold_3_days INTEGER,
    new_3_days INTEGER,
    median_7_day REAL,
    price_vol_7_day REAL,
    median_price_delta_7_day REAL,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


PRICE_QUERY = """
SELECT 
    p.Timestamp,
    e.URL
FROM 
    Prices p
JOIN 
    Events e ON p.EventID = e.EventID
LEFT JOIN 
    PriceRecords pr ON p.Timestamp = pr.Timestamp 
                    AND p.EventID = pr.EventID 
                    AND p.AttractionInstanceID = pr.AttractionInstanceID
WHERE 
    pr.Timestamp IS NULL;
"""

BACKUP_PRICE_QUERY = """
SELECT
    p.Timestamp,
    e.URL,
    p.EventID
FROM 
    Prices p
JOIN 
    Events e ON p.EventID = e.EventID
WHERE SUBSTR(p.Timestamp, 1, 8) = '20241214'
LIMIT 10;
"""

PRICE_ID_QUERY = """
SELECT 
    p.Timestamp,
    e.URL,
    p.EventID
FROM 
    Prices p
JOIN 
    Events e ON p.EventID = e.EventID
LEFT JOIN 
    PriceRecords pr ON p.Timestamp = pr.Timestamp 
                    AND p.EventID = pr.EventID 
                    AND p.AttractionInstanceID = pr.AttractionInstanceID
WHERE 
    pr.Timestamp IS NULL;
"""

INSERT_RECORD = """
INSERT OR IGNORE INTO PriceRecords (Timestamp, EventID, AttractionInstanceID)
SELECT 
    Timestamp,
    EventID,
    AttractionInstanceID
FROM 
    Prices;
"""

# Define the SQL update query
UPDATE_QUERY = """
    UPDATE Prices
    SET 
        Stan_Min_Price = ?,
        Stan_Max_Price = ?,
        Resl_Min_Price = ?,
        Resl_Max_Price = ?,
        Stan_Count = ?,
        Resl_Count = ?
    WHERE 
        Timestamp = ?;
"""


class DB_API:
    def __init__(self, db_file=None):
        """Initialize the database connection."""
        if db_file is None:
            db_file = DB_FILE
            
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
            # cur.execute(CREATE_VENUES_TABLE)
            # cur.execute(CREATE_ATTRACTIONS_TABLE)
            # cur.execute(CREATE_EVENTS_TABLE)
            # cur.execute(CREATE_ATTRACTION_INSTANCE_TABLE)
            # cur.execute(CREATE_PRICES_TABLE)
            # cur.execute(CREATE_PRICE_RECORDS_TABLE)
            # cur.execute(CREATE_TICKETS_TABLE)
            # cur.execute(CREATE_TICKET_PRICES_TABLE)
            # cur.execute(CREATE_SECTION_TABLE)
            # cur.execute(CREATE_SECTION_PRICES_TABLE)
            cur.execute(CREATE_SECTION_MARKET_DATA_TABLE)
            self.conn.commit()
            logger.info(f"Created tables")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
    
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

    def batch_insert_events(self, events_data):
        try:
            cur = self.conn.cursor()
            # Define the SQL query with INSERT OR IGNORE to avoid duplicates
            insert_query = """
                INSERT OR IGNORE INTO Events (EventID, VenueID, URL, AttractionID1, AttractionID2, sale_start, sale_end, name, event_start, timezone, segment, genre, subgenre, type, subtype)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """

            value_before = self.get_len(cur, "Events")
            # Use executemany for efficient batch insert
            cur.executemany(insert_query, events_data)
            value_after = self.get_len(cur, "Events")
            self.conn.commit()

            added_entries = value_after - value_before
            duplicates = len(events_data) - added_entries
            logger.info(f"{len(events_data)} events inserted ({duplicates} duplicates skipped).")
            logger.info(f"Events Table size increased from {value_before} entries to {value_after} entries")
        except Exception as e:
            logger.error(f"Failed to insert Events data: {e}")

    def batch_insert_venues(self, venues_data):
        try:
            cur = self.conn.cursor()
            # Define the SQL query with INSERT OR IGNORE to avoid duplicates
            insert_query = """
                INSERT OR IGNORE INTO Venues (VenueID, name, city, state, country)
                VALUES (?, ?, ?, ?, ?);
            """
            value_before = self.get_len(cur, "Venues")
            # Use executemany for efficient batch insert
            cur.executemany(insert_query, venues_data)
            value_after = self.get_len(cur, "Venues")
            self.conn.commit()

            added_entries = value_after - value_before
            duplicates = len(venues_data) - added_entries
            logger.info(f"{len(venues_data)} venues inserted ({duplicates} duplicates skipped).")
            logger.info(f"Venues Table size increased from {value_before} entries to {value_after} entries")
        except Exception as e:
            logger.error(f"Failed to insert Venues data: {e}")

    def batch_insert_prices(self, prices_data):
        try:
            cur = self.conn.cursor()
            # Define the SQL query with INSERT OR IGNORE to avoid duplicates
            insert_query = """
                INSERT OR IGNORE INTO Prices (Timestamp, EventID, AttractionInstanceID, Min_Price, Max_Price, Currency)
                VALUES (?, ?, ?, ?, ?, ?);
            """

            value_before = self.get_len(cur, "Prices")
            # Use executemany for efficient batch insert
            cur.executemany(insert_query, prices_data)
            value_after = self.get_len(cur, "Prices")
            self.conn.commit()

            added_entries = value_after - value_before
            duplicates = len(prices_data) - added_entries
            logger.info(f"{len(prices_data)} prices inserted ({duplicates} duplicates skipped).")
            logger.info(f"Prices Table size increased from {value_before} entries to {value_after} entries")
        except Exception as e:
            logger.error(f"Failed to insert Prices data: {e}")
    
    def batch_insert_attractions(self, attractions_data):
        try:
            cur = self.conn.cursor()
            # Define the SQL query with INSERT OR IGNORE to avoid duplicates
            insert_query = """
                INSERT OR IGNORE INTO Attractions (AttractionID, name)
                VALUES (?, ?);
            """
            
            value_before = self.get_len(cur, "Attractions")
            # Use executemany for efficient batch insert
            cur.executemany(insert_query, attractions_data)
            value_after = self.get_len(cur, "Attractions")
            self.conn.commit()

            added_entries = value_after - value_before
            duplicates = len(attractions_data) - added_entries
            logger.info(f"{len(attractions_data)} attractions inserted ({duplicates} duplicates skipped).")
            logger.info(f"Attractions Table size increased from {value_before} entries to {value_after} entries")
        except Exception as e:
            logger.error(f"Failed to insert Attractions data: {e}")
    
    def batch_insert_attraction_instances(self, attraction_instances_data):
        try:
            cur = self.conn.cursor()
            # Define the SQL query with INSERT OR IGNORE to avoid duplicates
            insert_query = """
                INSERT OR IGNORE INTO AttractionInstanceDetail (AttractionInstanceID, AttractionID1, Attraction1Rank, AttractionID2, Attraction2Rank)
                VALUES (?, ?, ?, ?, ?);
            """
            
            value_before = self.get_len(cur, "AttractionInstanceDetail")
            # Use executemany for efficient batch insert
            cur.executemany(insert_query, attraction_instances_data)
            value_after = self.get_len(cur, "AttractionInstanceDetail")
            self.conn.commit()

            added_entries = value_after - value_before
            duplicates = len(attraction_instances_data) - added_entries
            logger.info(f"{len(attraction_instances_data)} attraction instances inserted ({duplicates} duplicates skipped).")
            logger.info(f"Attraction Instances Table size increased from {value_before} entries to {value_after} entries")
        except Exception as e:
            logger.error(f"Failed to insert Attraction Instances data: {e}")
    
    def get_len(self, cur, table_name):
        query = f"SELECT COUNT(*) FROM {table_name};"
        cur.execute(query)
        row_count = cur.fetchone()[0]
        return row_count

    def get_event_name(self, event_id: str) -> str | None:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT name FROM Events WHERE EventID = ?", (event_id,))
            row = cur.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to fetch event name for EventID {event_id}: {e}")
            return None

    def get_prices(self):
        try:
            cur = self.conn.cursor()
            cur.execute(PRICE_QUERY)
            # Fetching all results
            results = cur.fetchall()
            # Getting column names from cursor
            colnames = [description[0] for description in cur.description]
            # Loading results into a DataFrame
            df = pd.DataFrame(results, columns=colnames)
            return df
        except Exception as e:
            logger.error(f"Failed to pull Price data: {e}")

    def get_event_ids_prices(self):
        try:
            cur = self.conn.cursor()
            cur.execute(PRICE_ID_QUERY)
            # Fetching all results
            results = cur.fetchall()
            # Getting column names from cursor
            colnames = [description[0] for description in cur.description]
            # Loading results into a DataFrame
            df = pd.DataFrame(results, columns=colnames)
            return df
        except Exception as e:
            logger.error(f"Failed to pull Price & ID data: {e}")

    def update_prices(self, df):
        try:
            cur = self.conn.cursor()
            for index, row in df.iterrows():
                cur.execute(
                    UPDATE_QUERY,
                    (
                        row['Stan_Min'],
                        row['Stan_Max'],
                        row['Resl_Min'],
                        row['Resl_Max'],
                        row['Stan_Count'],
                        row['Resl_Count'],
                        row['Timestamp']
                    )
                )
            self.conn.commit()
            logger.info(f"Updated {len(df)} elements of price data successfully")

        except Exception as e:
            logger.error(f"Failed to update Price data: {e}")

    def insert_records(self):
        try:
            cur = self.conn.cursor()
            cur.execute(INSERT_RECORD)
            self.conn.commit()
            logger.info("Inserted records")

        except Exception as e:
            logger.error(f"Failed to insert Price record: {e}")

    def upsert_tickets_and_prices(self, rows: list[dict]) -> None:
        cur = self.conn.cursor()
        ticket_value_before = self.get_len(cur, "Tickets")
        ticket_prices_value_before = self.get_len(cur, "TicketPrices")

        # Deduplicate to reduce conflict checks
        tickets_map = {}
        prices_map = {}

        for r in rows:
            tid = r.get("TicketID")
            if tid:
                tickets_map[tid] = (
                    r.get("EventID"),
                    r.get("Section"),
                    r.get("Row"),
                    r.get("Seat"),
                    tid,
                )

            tpid = r.get("TicketPriceID")
            if tpid:
                prices_map[tpid] = (
                    tpid,
                    r.get("TimestampUUID"),
                    r.get("TicketID"),
                    r.get("TicketPrice"),
                    r.get("Currency", "USD"),
                    r.get("SellableQuantitiesJson", "[]"),
                )

        tickets_params = list(tickets_map.values())
        prices_params = list(prices_map.values())

        # Single transaction for speed
        # Batch insert tickets
        cur.executemany(
            """
            INSERT INTO Tickets (EventID, Section, Row, Seat, TicketID)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(TicketID) DO NOTHING;
            """,
            tickets_params
        )

        # Batch insert ticket prices - skip if exists
        cur.executemany(
            """
            INSERT INTO TicketPrices (
                TicketPriceRowID, Timestamp, TicketID, TicketPrice, Currency, SellableQuantity
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(TicketPriceRowID) DO NOTHING;
            """,
            prices_params
        )

        ticket_value_after = self.get_len(cur, "Tickets")
        ticket_prices_value_after = self.get_len(cur, "TicketPrices")
        self.conn.commit()

        logger.info(
            f"Inserted {ticket_value_after - ticket_value_before} New Tickets, "
            f"Inserted {ticket_prices_value_after - ticket_prices_value_before} New Ticket Prices"
        )

    def populate_section_table(self):
        """
        Ensure each (EventID, Section) has a Section row.
        SectionID = EventID + '_' + Section
        """

        sql = """
            INSERT OR IGNORE INTO Section (SectionID, EventID, Section)
            SELECT
                EventID || '_' || Section AS SectionID,
                EventID,
                Section
            FROM Tickets
            WHERE Section IS NOT NULL
            GROUP BY EventID, Section;
            """
        cur = self.conn.cursor()
        section_value_before = self.get_len(cur, "Section")
        cur.execute(sql)
        self.conn.commit()
        section_value_after = self.get_len(cur, "Section")

        logger.info(
            f"Inserted {section_value_after - section_value_before} New Sections, "
        )

    def populate_section_prices(self):
        """
        For each Prices.Timestamp and Section, aggregate TicketPrices into SectionPrices.
        Currency assumed to ALWAYS be USD.
        """

        self.conn.create_aggregate("median", 1, Median)
        cur = self.conn.cursor()

        sql = """
        INSERT OR REPLACE INTO SectionPrices (
        SectionPriceRowID,
        Timestamp,
        SectionID,
        MedianPrice,
        AveragePrice,
        Currency,
        NumberOfTickets,
        MaxPrice,
        MinPrice,
        CreatedAt
        )
        SELECT
            p.Timestamp || '_' || s.SectionID AS SectionPriceRowID,
            p.Timestamp,
            s.SectionID,
            median(tp.TicketPrice) AS MedianPrice,
            AVG(tp.TicketPrice)    AS AveragePrice,
            'USD'                  AS Currency,
            COUNT(*)               AS NumberOfTickets,
            MAX(tp.TicketPrice)    AS MaxPrice,
            MIN(tp.TicketPrice)    AS MinPrice,
            CURRENT_TIMESTAMP      AS CreatedAt
        FROM Prices p
        JOIN TicketPrices tp
            ON tp.Timestamp = p.Timestamp
        JOIN Tickets t
            ON t.TicketID = tp.TicketID
        JOIN Section s
            ON s.EventID = t.EventID
           AND s.Section = t.Section
        WHERE NOT EXISTS (
                SELECT 1
                FROM PriceRecords pr
                WHERE pr.Timestamp = p.Timestamp
        )
        GROUP BY
            p.Timestamp,
            s.SectionID;
        """
        section_prices_value_before = self.get_len(cur, "SectionPrices")
        cur.execute(sql)
        self.conn.commit()
        section_prices_value_after = self.get_len(cur, "SectionPrices")

        logger.info(
            f"Inserted {section_prices_value_after - section_prices_value_before} New Section Prices, "
        )

    def pull_section_prices(self) -> pandas.DataFrame:
        conn = self.conn
        df_sp = pd.read_sql_query(
            """
            SELECT
                SectionPriceRowID,
                Timestamp,
                SectionID,
                MedianPrice
            FROM SectionPrices
            """,
            conn,
        )
        return df_sp

    def pull_ticket_ids(self, window):
        conn = self.conn
        df_tp = pd.read_sql_query(
            f"""
            SELECT 
                tp.Timestamp,
                t.TicketID,
                s.SectionID,
                sp.CreatedAt
            FROM TicketPrices tp
            JOIN Tickets t 
                ON t.TicketID = tp.TicketID
            JOIN Section s 
                ON s.EventID = t.EventID
               AND s.Section = t.Section
            JOIN SectionPrices sp
                ON sp.SectionID = s.SectionID
               AND sp.Timestamp = tp.Timestamp
            WHERE sp.CreatedAt >= datetime('now', '-{window} days');
            """,
            conn,
        )
        ticket_cache = (
            df_tp.groupby(["Timestamp", "SectionID"])["TicketID"]
            .apply(set)
            .to_dict()
        )

        del df_tp
        logger.info(f"Pulled {len(ticket_cache)} Section ticket groups")
        return ticket_cache

    def insert_section_market_data_record(self, df):
        cur = self.conn.cursor()

        if df.empty:
            logger.info("No SectionMarketData to insert: df is empty.")
            return

        # --- 1) We need SectionPriceRowID AND Timestamp from df ---
        # Ensure df contains Timestamp (it should come from df_daily)
        df_res = df[
            [
                "SectionPriceRowID",
                "Timestamp",  # <-- REQUIRED FOR FILTERING
                "sold_3_days",
                "new_3_days",
                "median_7_day",
                "price_vol_7_day",
                "median_price_delta_7_day",
            ]
        ].copy()

        # --- 2) Load all timestamps already in PriceRecords ---
        cur.execute("SELECT DISTINCT Timestamp FROM PriceRecords;")
        ts_in_pr = {row[0] for row in cur.fetchall()}

        # --- 3) Build params only for SectionPrices whose Timestamp is NOT in PriceRecords ---
        params = []
        skipped = 0

        def to_int(x):
            return None if pd.isna(x) else int(x)

        def to_float(x):
            return None if pd.isna(x) else float(x)

        for _, r in df_res.iterrows():
            spid = r["SectionPriceRowID"]
            ts = r["Timestamp"]

            # Skip if timestamp already present in PriceRecords
            if ts in ts_in_pr:
                skipped += 1
                continue

            params.append(
                (
                    f"{spid}_mark",  # SectionMarketDataID
                    spid,  # SectionPriceRowID
                    to_int(r["sold_3_days"]),
                    to_int(r["new_3_days"]),
                    to_float(r["median_7_day"]),
                    to_float(r["price_vol_7_day"]),
                    to_float(r["median_price_delta_7_day"]),
                )
            )

        # No new rows → done
        if not params:
            logger.info(f"No new SectionMarketData rows to upsert (skipped {skipped}).")
            return

        # --- 4) Execute the INSERT OR REPLACE ---
        cur.executemany(
            """
            INSERT OR REPLACE INTO SectionMarketData (
                SectionMarketDataID,
                SectionPriceRowID,
                sold_3_days,
                new_3_days,
                median_7_day,
                price_vol_7_day,
                median_price_delta_7_day
            )
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            params,
        )

        self.conn.commit()
        logger.info(f"Upserted {len(params)} new SectionMarketData rows, skipped {skipped} already in PriceRecords.")

    def pull_ticket_features(self, feature_query, batch_size, offset) -> pandas.DataFrame:
        self.conn.create_function("to_sqlite_ts", 1, build_sqlite_timestamp)
        df = pd.read_sql_query(
            feature_query,
            self.conn,
            params=(batch_size, offset),
        )
        return df



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
            os.makedirs(DB_BACKUP, exist_ok=True)

            # Extract base DB filename
            base_name = os.path.basename(DB_FILE).replace(".db", "")

            # Construct backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{base_name}_backup_{timestamp}.db"

            backup_path = os.path.join(DB_BACKUP, backup_filename)

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


def insert_and_backup():
    db = DB_API()
    db.insert_records()
    db.backup_database()
    db.close()
