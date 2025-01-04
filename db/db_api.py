import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from decimal import Decimal
import json
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger

DB_HOST = os.getenv('DB_HOST')
DB_NAME = 'postgres'   # Database name
DB_UNAME = 'postgres'   # Username
DB_PWD = os.getenv('DB_PWD')
DB_PORT = 5432

# SQL statements to create tables
CREATE_VENUES_TABLE = """
CREATE TABLE IF NOT EXISTS Venues (
    VenueID VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255),
    city VARCHAR(50),
    state VARCHAR(10),
    country VARCHAR(10)
);
"""

CREATE_ATTRACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS Attractions (
    AttractionID VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255)
);
"""

CREATE_ATTRACTION_INSTANCE_TABLE = """
CREATE TABLE IF NOT EXISTS AttractionInstanceDetail (
    AttractionInstanceID VARCHAR(100) PRIMARY KEY,
    AttractionID1 VARCHAR(50) REFERENCES Attractions(AttractionID),
    Attraction1Rank INT,
    AttractionID2 VARCHAR(50) REFERENCES Attractions(AttractionID),
    Attraction2Rank INT
);
"""

CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS Events (
    EventID VARCHAR(50) PRIMARY KEY,
    VenueID VARCHAR(50) REFERENCES Venues(VenueID),
    URL VARCHAR(400),
    AttractionID1 VARCHAR(50) REFERENCES Attractions(AttractionID),
    AttractionID2 VARCHAR(50) REFERENCES Attractions(AttractionID),
    sale_start TIMESTAMP,
    sale_end TIMESTAMP,
    name VARCHAR(255),
    event_start TIMESTAMP,
    timezone VARCHAR(100),
    segment VARCHAR(100),
    genre VARCHAR(100),
    subgenre VARCHAR(100),
    type VARCHAR(100),
    subtype VARCHAR(100)
);
"""

CREATE_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS Prices (
    Timestamp VARCHAR(100) PRIMARY KEY,
    EventID VARCHAR(50) REFERENCES Events(EventID),
    AttractionInstanceID VARCHAR(100) REFERENCES AttractionInstanceDetail(AttractionInstanceID),
    Min_Price DECIMAL,
    Max_Price DECIMAL,
    Currency VARCHAR(10),
    Stan_Min_Price DECIMAL,
    Stan_Max_Price DECIMAL,
    Resl_Min_Price DECIMAL,
    Resl_Max_Price DECIMAL
);
"""

CREATE_PRICE_RECORDS_TABLE = """
CREATE TABLE IF NOT EXISTS PriceRecords (
    Timestamp VARCHAR(100),
    EventID VARCHAR(50),
    AttractionInstanceID VARCHAR(100),
    PRIMARY KEY (Timestamp, EventID, AttractionInstanceID)
);
"""

CREATE_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS AuditTable (
    Timestamp VARCHAR(100) PRIMARY KEY,
    DatePulled TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
WHERE SUBSTRING(p.Timestamp, 1, 8) = '20241214'
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

DROP_ATTRACTIONS = """
DELETE FROM
    AttractionInstanceDetail a
WHERE SUBSTRING(a.AttractionInstanceID, 1, 8) = '20241122';
"""

INSERT_RECORD = """
INSERT INTO PriceRecords (Timestamp, EventID, AttractionInstanceID)
SELECT 
    Timestamp,
    EventID,
    AttractionInstanceID
FROM 
    Prices
ON CONFLICT (Timestamp, EventID, AttractionInstanceID) DO NOTHING;
"""

# Define the SQL update query
UPDATE_QUERY = """
    UPDATE Prices
    SET 
        Stan_Min_Price = %s,
        Stan_Max_Price = %s,
        Resl_Min_Price = %s,
        Resl_Max_Price = %s
    WHERE 
        Timestamp = %s;
"""





class DB_API:
    def __init__(self):
        """Initialize the database connection."""
        try:
            logger.info("Database connection establishing....")
            self.conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_UNAME,
                password=DB_PWD,
                port=DB_PORT
            )
            self.conn.autocommit = True  # Enable auto-commit mode for simplicity
            logger.info("Database connection established.")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            self.conn = None
    
    def create_tables(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute(CREATE_VENUES_TABLE)
                cur.execute(CREATE_ATTRACTIONS_TABLE)
                cur.execute(CREATE_EVENTS_TABLE)
                cur.execute(CREATE_ATTRACTION_INSTANCE_TABLE)
                cur.execute(CREATE_PRICES_TABLE)
                cur.execute(CREATE_PRICE_RECORDS_TABLE)
                logger.info(f"Created tables")
        except Exception as e:
            logger.error(f"Failed to create tables")
    
    def drop_tables(self):
        try:
            with self.conn.cursor() as cur:
                # Get the list of all tables in the public schema
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public';
                """)
                tables = cur.fetchall()
                # Drop each table
                for table in tables:
                    table_name = table[0]
                    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                    logger.info(f"Table {table_name} dropped successfully.")

                logger.info("All tables deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")

    def drop_attraction_instance(self):
        try:
            with self.conn.cursor() as cur:
                value_before = self.get_len(cur, "AttractionInstanceDetail")
                cur.execute(DROP_ATTRACTIONS)
                value_after = self.get_len(cur, "AttractionInstanceDetail")
            deleted_entries = value_before - value_after
            logger.info(f"{deleted_entries} attraction instances deleted")

        except Exception as e:
            logger.error(f"Failed to delete data: {e}")


    def batch_insert_events(self, events_data):
        try:
            with self.conn.cursor() as cur:
                # Define the SQL query with ON CONFLICT DO NOTHING to avoid duplicates
                insert_query = """
                    INSERT INTO Events (EventID, VenueID, URL, AttractionID1, AttractionID2, sale_start, sale_end, name, event_start, timezone, segment, genre, subgenre, type, subtype)
                    VALUES %s
                    ON CONFLICT (EventID) DO NOTHING;
                """

                value_before = self.get_len(cur, "Events")
                # Use execute_values for efficient batch insert
                execute_values(cur, insert_query, events_data)
                value_after = self.get_len(cur, "Events")

            added_entries = value_after - value_before
            duplicates = len(events_data) - added_entries
            logger.info(f"{len(events_data)} events inserted ({duplicates} duplicates skipped).")
            logger.info(f"Events Table size increased from {value_before} entries to {value_after} entries")
        except Exception as e:
            logger.error(f"Failed to insert Events data: {e}")

    def batch_insert_venues(self, venues_data):
        try:
            with self.conn.cursor() as cur:
                # Define the SQL query with ON CONFLICT DO NOTHING to avoid duplicates
                insert_query = """
                    INSERT INTO Venues (VenueID, name, city, state, country)
                    VALUES %s
                    ON CONFLICT (VenueID) DO NOTHING;
                """
                value_before = self.get_len(cur, "Venues")
                # Use execute_values for efficient batch insert
                execute_values(cur, insert_query, venues_data)
                value_after = self.get_len(cur, "Venues")

            added_entries = value_after - value_before
            duplicates = len(venues_data) - added_entries
            logger.info(f"{len(venues_data)} venues inserted ({duplicates} duplicates skipped).")
            logger.info(f"Venues Table size increased from {value_before} entries to {value_after} entries")
        except Exception as e:
            logger.error(f"Failed to insert Venues data: {e}")

    def batch_insert_prices(self, prices_data):
        try:
            with self.conn.cursor() as cur:
                # Define the SQL query with ON CONFLICT DO NOTHING to avoid duplicates
                insert_query = """
                    INSERT INTO Prices (Timestamp, EventID, AttractionInstanceID, Min_Price, Max_Price, Currency)
                    VALUES %s
                    ON CONFLICT (Timestamp) DO NOTHING;
                """

                value_before = self.get_len(cur, "Prices")
                # Use execute_values for efficient batch insert
                execute_values(cur, insert_query, prices_data)
                value_after = self.get_len(cur, "Prices")

            added_entries = value_after - value_before
            duplicates = len(prices_data) - added_entries
            logger.info(f"{len(prices_data)} prices inserted ({duplicates} duplicates skipped).")
            logger.info(f"Prices Table size increased from {value_before} entries to {value_after} entries")
        except Exception as e:
            logger.error(f"Failed to insert Prices data: {e}")
    
    def batch_insert_attractions(self, attractions_data):
        try:
            with self.conn.cursor() as cur:
                # Define the SQL query with ON CONFLICT DO NOTHING to avoid duplicates
                insert_query = """
                    INSERT INTO Attractions (AttractionID, name)
                    VALUES %s
                    ON CONFLICT (AttractionID) DO NOTHING;
                """
                
                value_before = self.get_len(cur, "Attractions")
                # Use execute_values for efficient batch insert
                execute_values(cur, insert_query, attractions_data)
                value_after = self.get_len(cur, "Attractions")

            added_entries = value_after - value_before
            duplicates = len(attractions_data) - added_entries
            logger.info(f"{len(attractions_data)} attractions inserted ({duplicates} duplicates skipped).")
            logger.info(f"Attractions Table size increased from {value_before} entries to {value_after} entries")
        except Exception as e:
            logger.error(f"Failed to insert Attractions data: {e}")
    
    def batch_insert_attraction_instances(self, attraction_instances_data):
        try:
            with self.conn.cursor() as cur:
                # Define the SQL query with ON CONFLICT DO NOTHING to avoid duplicates
                insert_query = """
                    INSERT INTO AttractionInstanceDetail (AttractionInstanceID, AttractionID1, Attraction1Rank, AttractionID2, Attraction2Rank)
                    VALUES %s
                    ON CONFLICT (AttractionInstanceID) DO NOTHING;
                """
                
                value_before = self.get_len(cur, "AttractionInstanceDetail")
                # Use execute_values for efficient batch insert
                execute_values(cur, insert_query, attraction_instances_data)
                value_after = self.get_len(cur, "AttractionInstanceDetail")

            added_entries = value_after - value_before
            duplicates = len(attraction_instances_data) - added_entries
            logger.info(f"{len(attraction_instances_data)} attraction instances inserted ({duplicates} duplicates skipped).")
            logger.info(f"Attraction Instances Table size increased from {value_before} entries to {value_after} entries")
        except Exception as e:
            logger.error(f"Failed to insert Attraction Instances data: {e}")


    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")
    
    def get_len(self, cur, table_name):
        query = f"SELECT COUNT(*) FROM {table_name};"
        cur.execute(query)
        row_count = cur.fetchone()[0]
        return row_count

    def get_prices(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute(PRICE_QUERY)
                # Fetching all results
                results = cur.fetchall()
                # Getting column names from cursor
                colnames = [desc[0] for desc in cur.description]
                # Loading results into a DataFrame
                df = pd.DataFrame(results, columns=colnames)
            return df
        except Exception as e:
            logger.error(f"Failed to pull Price data: {e}")

    def get_event_ids_prices(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute(PRICE_ID_QUERY)
                # Fetching all results
                results = cur.fetchall()
                # Getting column names from cursor
                colnames = [desc[0] for desc in cur.description]
                # Loading results into a DataFrame
                df = pd.DataFrame(results, columns=colnames)
            return df
        except Exception as e:
            logger.error(f"Failed to pull Price & ID data: {e}")

    def insert_records(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute(INSERT_RECORD)

            logger.info("Inserted records")
                
        except Exception as e:
            logger.error(f"Failed to insert Price record: {e}")

    def update_prices(self, df):
        try:
            with self.conn.cursor() as cur:
                for index, row in df.iterrows():
                    cur.execute(
                        UPDATE_QUERY,
                        (
                            row['Stan_Min'],
                            row['Stan_Max'],
                            row['Resl_Min'],
                            row['Resl_Max'],
                            row['timestamp']
                        )
                    )
            logger.info(f"Updated {len(df)} elements of price data successfully")


        except Exception as e:
            logger.error(f"Failed to insert Price data: {e}")
    
    # def fetch_event_prices(self):
    #     event_prices_df = pd.read_sql(price_query, self.conn)
    #
    #     return event_prices_df
    def get_train_data(self, query):
        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
                # Fetching all results
                results = cur.fetchall()
                # Getting column names from cursor
                colnames = [desc[0] for desc in cur.description]
                # Loading results into a DataFrame
                df = pd.DataFrame(results, columns=colnames)
            return df
        except Exception as e:
            logger.error(f"Failed to pull Train data: {e}")

    def fetch_as_json(self, table_name):
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        """
        Fetches all rows from a specified table and returns the result as a JSON string.
        
        Parameters:
            table_name (str): The name of the table to fetch data from.
        
        Returns:
            str: JSON-formatted string of the table's contents.
        """
        if not self.conn:
            logger.info("No database connection.")
            return None

        try:
            with self.conn.cursor() as cur:
                query = f"SELECT * FROM {table_name}"
                cur.execute(query)
                rows = cur.fetchall()
                column_names = [desc[0] for desc in cur.description]
                result = [dict(zip(column_names, row)) for row in rows]
                # Convert result to JSON
                json_result = json.dumps(result, default=convert_datetime, indent=4)
                return json_result  # Convert to JSON format
        except Exception as e:
            logger.info(f"Failed to fetch data from {table_name}: {e}")
            return None
