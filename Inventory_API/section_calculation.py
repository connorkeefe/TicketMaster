from logger import logger
from db.db_api import DB_API
from datetime import datetime, timedelta
import pandas as pd
import tqdm

def populate_section_market_data(db):
    """
    For each SectionPrices row, compute:
      - sold_3_days, new_3_days (based on TicketIDs)
      - median_7_day (rolling 7-calendar-day median of MedianPrice)
      - price_vol_7_day (rolling 7-calendar-day std dev of MedianPrice)
      - median_price_delta_7_day = MedianPrice - median_7_day

    Assumptions / rules:
      - Timestamp format is 'YYYYMMDDHHMMSS-uuid' (TimestampUUID).
        We take the first 8 chars (YYYYMMDD) as the 'day'.
      - For sold_3_days / new_3_days we require *consecutive calendar days*.
          * For a given day D0, we require data for day D0-1.
          * If we ALSO have data for D0-2, that's a full 3-day window.
          * If D0-1 is missing, we DO NOT calculate sold/new (set NULL).
      - sold_3_days / new_3_days are computed between the
        earliest available day in that consecutive window and the current day:
          * sold_3_days = #TicketIDs present on earliest day but not on current day
          * new_3_days  = #TicketIDs present on current day but not on earliest day
      - SectionMarketDataID is set equal to SectionPriceRowID so we have
        exactly one market-data record per SectionPrices row.
    """
    logger.info("Starting Section Market Data Calculations and Insert")
    # ------------------------------------------------------------------
    # 1) Pull SectionPrices (one row per Timestamp + SectionID)
    # ------------------------------------------------------------------
    df_sp = db.pull_section_prices()

    logger.info(f"Pulled {len(df_sp)} rows from section prices")


    if len(df_sp) == 0:
        logger.info("No SectionPrices rows found; nothing to populate for SectionMarketData.")
        return

    # Extract calendar day from TimestampUUID (YYYYMMDD part)
    # Extract full datetime from TimestampUUID
    dt = pd.to_datetime(df_sp["Timestamp"].str.slice(0, 14), format="%Y%m%d%H%M%S")

    # If the time is between 00:00:00 and 04:59:59, shift to previous day
    df_sp["date"] = dt.where(dt.dt.hour >= 10, dt - pd.Timedelta(days=1))
    df_sp["date"] = df_sp["date"].dt.normalize()  # Keep only YYYY-MM-DD

    days_back_section = 7
    # Find latest date we have
    max_date = df_sp["date"].max()
    cutoff = max_date - pd.Timedelta(days=days_back_section - 1)
    df_sp = df_sp[df_sp["date"] >= cutoff]
    # Sort for deterministic grouping
    df_sp = df_sp.sort_values(["SectionID", "date", "Timestamp"])
    logger.info(f"Filtered to recent date window, now {len(df_sp)} rows ")

    # If there are multiple snapshots per day per SectionID, keep the latest one
    df_daily = (
        df_sp.groupby(["SectionID", "date"], as_index=False)
        .tail(1)  # last row per (SectionID, date)
        .reset_index(drop=True)
    )
    logger.info(f"Trimmed df by time duplicate is {len(df_daily)} rows")

    # ------------------------------------------------------------------
    # 2) Rolling 7-day metrics on MedianPrice (calendar-time-based)
    # ------------------------------------------------------------------
    def add_rolling_metrics(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").set_index("date")

        # 7 calendar-day window, based on index
        group["median_7_day"] = (
            group["MedianPrice"].rolling("7D", min_periods=1).median()
        )
        group["price_vol_7_day"] = (
            group["MedianPrice"].rolling("7D", min_periods=2).std()
        )
        group["median_price_delta_7_day"] = (
                group["MedianPrice"] - group["median_7_day"]
        )

        return group.reset_index()

    df_daily = (
        df_daily.groupby("SectionID", group_keys=False)
        .apply(add_rolling_metrics)
    )
    df_daily = df_daily.reset_index(drop=False)

    # ------------------------------------------------------------------
    # 3) sold_3_days / new_3_days computation using TicketIDs
    # ------------------------------------------------------------------
    window_days = 6  # search for consecutive pairs within this many days back from date0
    ticket_cache = db.pull_ticket_ids(window_days + 1)

    df_daily["sold_3_days"] = pd.NA
    df_daily["new_3_days"] = pd.NA

    for section_id, g in tqdm.tqdm(df_daily.groupby("SectionID")):
        # Work on a date-sorted view for this section
        g = g.sort_values("date")

        # We’ll use the original df_daily index to write results back
        for idx, row in g.iterrows():
            # logger.info(idx)
            date0 = row["date"]
            # 1) Limit to days in the [date0 - window_days, date0] window
            window_start = date0 - timedelta(days=window_days)
            mask = (g["date"] >= window_start) & (g["date"] <= date0)
            days = g.loc[mask, "date"].drop_duplicates().sort_values()

            # 2) Find all consecutive-day pairs (d_prev, d_next) within the window
            pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            prev_day = None
            for d in days:
                if prev_day is not None and d - prev_day == timedelta(days=1):
                    pairs.append((prev_day, d))
                prev_day = d

            # No consecutive days → cannot compute sold/new
            if not pairs or len(pairs) == 1:
                # logger.info(f"Skipping improper pair: {pairs}")
                continue

            # 3) Take the most recent up to 2 pairs (based on the later day in the pair)
            pairs.sort(key=lambda p: p[1], reverse=True)
            selected_pairs = pairs[:2]

            total_sold = 0
            total_new = 0

            # 4) For each pair, compute sold/new and accumulate
            for d_prev, d_next in selected_pairs:
                # Each (SectionID, date) has exactly one row in df_daily
                ts_prev = g.loc[g["date"] == d_prev, "Timestamp"].iloc[-1]
                ts_next = g.loc[g["date"] == d_next, "Timestamp"].iloc[-1]
                # if key in ticket_cache:
                tickets_prev = ticket_cache[(ts_prev, section_id)]
                tickets_next = ticket_cache[(ts_next, section_id)]

                # Unique TicketIDs: sold = present then gone; new = absent then present
                sold_pair = len(tickets_prev - tickets_next)
                new_pair = len(tickets_next - tickets_prev)
                total_sold += sold_pair
                total_new += new_pair

            # logger.info(f"Adding sold and new values {total_sold}, {total_new}")
            # 5) Write back to df_daily for this date0 row
            df_daily.at[idx, "sold_3_days"] = total_sold
            df_daily.at[idx, "new_3_days"] = total_new

    # ------------------------------------------------------------------
    # 4) Write results to SectionMarketData (upsert, 1 row per SectionPriceRowID)
    # ------------------------------------------------------------------

    db.insert_section_market_data_record(df_daily)

