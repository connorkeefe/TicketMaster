from db.db_api import DB_API
from db.db_api_ml import DB_API_ML, build_sqlite_timestamp
from logger import logger
from db.db_api import insert_and_backup
from datetime import datetime, timedelta
import pandas as pd
import sqlite3

def migrate_add_timestampnorm():
    db = DB_API_ML()
    conn = db.conn
    cur = conn.cursor()

    # 1) Add the column if it doesn't exist yet
    try:
        cur.execute("ALTER TABLE TicketDailyFeatures ADD COLUMN TimestampNorm TIMESTAMP;")
        conn.commit()
        logger.info("Added TimestampNorm column to TicketDailyFeatures.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            logger.info("TimestampNorm column already exists, skipping ALTER TABLE.")
        else:
            raise

    # 2) Backfill the column for existing rows
    # We'll do it in Python to reuse the exact build_sqlite_timestamp logic
    cur.execute(f"""
            SELECT TicketDailyFeatureID, Timestamp
            FROM TicketDailyFeatures
            WHERE Timestamp IS NOT NULL
              AND (TimestampNorm IS NULL OR TimestampNorm = '');
        """)
    rows = cur.fetchall()
    logger.info(f"Rows to backfill TimestampNorm: {len(rows):,}")

    update_sql = "UPDATE TicketDailyFeatures SET TimestampNorm = ? WHERE TicketDailyFeatureID = ?;"
    batch = []


    for tid, ts_raw in rows:
        ts_norm = build_sqlite_timestamp(ts_raw)
        batch.append((ts_norm, tid))

    cur.executemany(update_sql, batch)
    conn.commit()
    logger.info(f"Backfilled {len(batch)} rows...")
    batch.clear()

    db.close()
    logger.info("Migration complete.")


def build_ticket_daily_features(
    incremental: bool = False,
    batch_size: int = 100_000,
):
    """
    Build the TicketDailyFeatures table in the ML database by querying the main DB.

    - incremental=False: scan ALL TicketPrices
    - incremental=True: only include rows where associated Prices row is NOT in PriceRecords
      (using LEFT JOIN on PriceRecords)

    Uses LIMIT/OFFSET batching to keep memory reasonable for ~29M TicketPrices.
    """

    # --- open main DB (events_database.db) ---
    event_db = DB_API()
    # --- open / prepare ML DB (ml_database.db) ---
    ml_db = DB_API_ML()

    # --- base query for features ---
    # IMPORTANT:
    #  - we ALWAYS left join PriceRecords (pr)
    #  - when incremental=True, we filter to pr.Timestamp IS NULL (not in PriceRecords)
    #  - when incremental=False, we use WHERE 1=1 (no filter)
    where_predicate = "pr.Timestamp IS NULL" if incremental else "1=1"

    events_table_feature_query = f"""
    SELECT
        tp.TicketPriceRowID                    AS TicketPriceRowID,
        tp.TicketID                            AS TicketID,
        t.EventID                              AS EventID,
        s.SectionID                            AS SectionID,
        tp.Timestamp                           AS Timestamp,

        e.name                                 AS event_name,
        a1.name                                AS attraction_name_1,
        a2.name                                AS attraction_name_2,
        v.name                                 AS venue_name,
        e.segment                              AS segment,
        e.genre                                AS genre,
        e.subgenre                             AS subgenre,
        v.city                                 AS venue_city,
        v.state                                AS venue_state,
        v.country                              AS venue_country,
        strftime('%w', e.event_start)         AS event_day_of_week,

        t.Section                              AS Section,
        t.Row                                  AS Row,
        t.Seat                                 AS Seat,

        CAST(julianday(to_sqlite_ts(tp.Timestamp)) - julianday(e.sale_start) AS INTEGER) AS sale_days_elapsed,
        CAST(julianday(e.event_start) - julianday(to_sqlite_ts(tp.Timestamp)) AS INTEGER) AS days_until_event,
        strftime('%w', to_sqlite_ts(tp.Timestamp))  AS day_of_week,
        aid.Attraction1Rank                    AS attraction_1_rank,
        aid.Attraction2Rank                    AS attraction_2_rank,

        tp.TicketPrice                         AS ticket_price,
        sp.NumberOfTickets                     AS section_ticket_count,
        sp.MinPrice                            AS section_min_price,
        sp.MaxPrice                            AS section_max_price,
        sp.AveragePrice                        AS section_avg_price,
        sp.MedianPrice                         AS section_median_price,
        smd.median_7_day                       AS section_7day_median,
        smd.price_vol_7_day                    AS section_7day_price_vol,
        smd.median_price_delta_7_day           AS section_7day_median_price_delta,
        CASE
            WHEN sp.MedianPrice IS NULL OR sp.MedianPrice = 0 THEN NULL
            ELSE (sp.MedianPrice - tp.TicketPrice) / sp.MedianPrice
        END                                    AS pct_below_section_median,
        smd.sold_3_days                        AS section_3days_sold,
        smd.new_3_days                         AS section_3days_new

    FROM TicketPrices tp
    JOIN Tickets t
        ON t.TicketID = tp.TicketID
    JOIN Prices p
        ON p.Timestamp = tp.Timestamp
    JOIN Events e
        ON e.EventID = t.EventID
    LEFT JOIN AttractionInstanceDetail aid
        ON aid.AttractionInstanceID = p.AttractionInstanceID
    LEFT JOIN Attractions a1
        ON a1.AttractionID = aid.AttractionID1
    LEFT JOIN Attractions a2
        ON a2.AttractionID = aid.AttractionID2
    LEFT JOIN Venues v
        ON v.VenueID = e.VenueID
    JOIN Section s
        ON s.EventID = t.EventID
       AND s.Section = t.Section
    JOIN SectionPrices sp
        ON sp.SectionID = s.SectionID
       AND sp.Timestamp = tp.Timestamp
    LEFT JOIN SectionMarketData smd
        ON smd.SectionPriceRowID = sp.SectionPriceRowID
    LEFT JOIN PriceRecords pr
        ON pr.Timestamp = p.Timestamp
       AND pr.EventID = p.EventID
       AND pr.AttractionInstanceID = p.AttractionInstanceID

    WHERE {where_predicate}
    LIMIT ? OFFSET ?;
    """

    logger.info(
        f"Starting TicketDailyFeatures build. "
        f"Incremental={incremental}, batch_size={batch_size}"
    )

    offset = 0
    batch_idx = 0
    total_rows = 0
    batch_needed = True
    try:
        while batch_needed:
            df_batch = event_db.pull_ticket_features(feature_query=events_table_feature_query, batch_size=batch_size, offset=offset)

            if df_batch.empty:
                logger.info("No rows returned from main DB; pipeline complete.")
                batch_needed = False
                continue

            batch_idx += 1
            logger.info(
                f"Fetched batch {batch_idx} from main DB "
                f"(offset={offset}, rows={len(df_batch)})"
            )

            # Insert into ML DB
            ml_db.insert_ticket_daily_features_batch(df_batch)
            total_rows += len(df_batch)

            if len(df_batch) < batch_size:
                batch_needed = False
            # Free memory before next batch
            del df_batch

            offset += batch_size

        logger.info(f"TicketDailyFeatures build complete. Total rows inserted (attempted): {total_rows}")

    except Exception as e:
        logger.info(f"Error building Daily Ticket features table: {e}")

    fill_ticketdaily_labels(ml_db)
    ml_db.backup_database()
    event_db.close()
    ml_db.close()
    logger.info("Closed both main and ML DB connections.")

def fill_ticketdaily_labels(db):
    conn = db.conn
    cur = conn.cursor()

    total = cur.execute("""
        SELECT COUNT(*)
        FROM TicketDailyFeatures
        WHERE label_7day_price IS NULL
           OR label_14day_price IS NULL;
    """).fetchone()[0]
    logger.info(f"Label-fill required rows: {total:,}")

    if total == 0:
        logger.info("Nothing to do, all labels already filled.")
        db.close()
        return

    # 1) 7-day ahead labels using TimestampNorm
    logger.info("Filling 7-day future prices and section medians...")
    cur.execute("""
        UPDATE TicketDailyFeatures AS t
        SET
            label_7day_section_median_price = (
                SELECT tf.section_median_price
                FROM TicketDailyFeatures AS tf
                WHERE tf.SectionID = t.SectionID
                  AND tf.TimestampNorm IS NOT NULL
                  AND date(tf.TimestampNorm) = date(t.TimestampNorm, '+7 days')
                LIMIT 1
            ),
            label_7day_price = (
                SELECT tf.ticket_price
                FROM TicketDailyFeatures AS tf
                WHERE tf.TicketID = t.TicketID
                  AND tf.TimestampNorm IS NOT NULL
                  AND date(tf.TimestampNorm) = date(t.TimestampNorm, '+7 days')
                LIMIT 1
            )
        WHERE (t.label_7day_price IS NULL
            OR t.label_7day_section_median_price IS NULL)
          AND t.TimestampNorm IS NOT NULL;
    """)
    logger.info(f"7-day fields updated, total_changes={conn.total_changes:,}")

    # 2) 14-day ahead labels using TimestampNorm
    logger.info("Filling 14-day future prices...")
    cur.execute("""
        UPDATE TicketDailyFeatures AS t
        SET
            label_14day_price = (
                SELECT tf.ticket_price
                FROM TicketDailyFeatures AS tf
                WHERE tf.TicketID = t.TicketID
                  AND tf.TimestampNorm IS NOT NULL
                  AND date(tf.TimestampNorm) = date(t.TimestampNorm, '+14 days')
                LIMIT 1
            )
        WHERE t.label_14day_price IS NULL
          AND t.TimestampNorm IS NOT NULL;
    """)
    logger.info(f"14-day fields updated, total_changes={conn.total_changes:,}")

    # 3) Compute returns
    logger.info("Computing 7- and 14-day returns...")
    cur.execute("""
        UPDATE TicketDailyFeatures
        SET
            label_7day_section_median_return =
                CASE
                    WHEN section_median_price IS NOT NULL
                     AND label_7day_section_median_price IS NOT NULL THEN
                        (label_7day_section_median_price - section_median_price)
                        / section_median_price
                END,

            label_7day_return =
                CASE
                    WHEN ticket_price IS NOT NULL
                     AND label_7day_price IS NOT NULL THEN
                        (label_7day_price - ticket_price) / ticket_price
                END,

            label_14day_return =
                CASE
                    WHEN ticket_price IS NOT NULL
                     AND label_14day_price IS NOT NULL THEN
                        (label_14day_price - ticket_price) / ticket_price
                END
        WHERE label_7day_price IS NOT NULL
           OR label_14day_price IS NOT NULL;
    """)
    logger.info(f"Return fields updated, total_changes={conn.total_changes:,}")

    conn.commit()
    db.close()
    logger.info("Label filling complete.")



def ml_daily_run():
    build_ticket_daily_features(incremental=True, batch_size=10_000_000)
    logger.info("Completed Daily ticket feature build")
    insert_and_backup()

if __name__ == "__main__":
    fill_ticketdaily_labels()
    ml_db = DB_API_ML()
    ml_db.backup_database()
    ml_db.close()
    #
    # migrate_add_timestampnorm()