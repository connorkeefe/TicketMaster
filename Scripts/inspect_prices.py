#!/usr/bin/env python3
"""
Event Price Daily Analysis (Coverage + Ticket Stats + Resale Stats)
------------------------------------------------------------------
Outputs: analysis_out/event_coverage.csv with per-EventID metrics:

From Prices.csv:
- row_count: # rows in Prices.csv for the EventID
- first_date, last_date, days_span
- completeness_ratio
- extraction_count, extraction_pct
- resale_value_count
- resale_variability, resale_variability_tiebreak, n_resale_rows
- resale_min_price, resale_max_price  (Resl_* from Prices)

From DB (Tickets + TicketPrices via DB_API / events_database.db):
- ticket_count: # distinct tickets in Tickets
- ticketprice_count: # TicketPrices rows (via join on TicketID)
- ticket_min_price, ticket_max_price: min/max TicketPrice

Also:
- Top-20 standardized plots (Resl_Min/Max/Count scaled 0-1)
- (plot_resale_min_max function is present & ready if you want it)

Usage:
  Adjust `prices_csv` path in main() if needed, then run.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from db.db_api import DB_API

# --- define extraction-related columns once ---
EXTRACTION_COLS = [
    "Stan_Min_Price", "Stan_Max_Price", "Stan_Count",
    "Resl_Min_Price", "Resl_Max_Price", "Resl_Count"
]

# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def read_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Timestamp" not in df.columns:
        raise ValueError("Missing 'Timestamp' column.")
    if "EventID" not in df.columns:
        raise ValueError("Missing 'EventID' column.")

    # Parse timestamp from first 14 chars (YYYYMMDDhhmmss)
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"].astype(str).str[:14],
        format="%Y%m%d%H%M%S",
        errors="coerce"
    )

    # Normalize extraction-related numeric columns
    for c in EXTRACTION_COLS:
        if c in df.columns:
            df[c] = df[c].replace(r"^\s*$", np.nan, regex=True)
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["EventID", "Timestamp"]).reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def minmax01(s: pd.Series) -> pd.Series:
    """Scale a numeric series to [0,1]. Handles constant/all-NaN gracefully."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s
    vmin = s.min()
    vmax = s.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return s * np.nan
    if np.isclose(vmin, vmax):
        return pd.Series(0.5, index=s.index)
    return (s - vmin) / (vmax - vmin)

# ---------------------------------------------------------------------------
# Coverage & extraction metrics (from Prices.csv)
# ---------------------------------------------------------------------------

def compute_event_coverage(df: pd.DataFrame) -> pd.DataFrame:
    g_ts = df.groupby("EventID")["Timestamp"]
    coverage = g_ts.agg(
        row_count="count",
        first_date="min",
        last_date="max"
    ).reset_index()

    coverage["days_span"] = (coverage["last_date"] - coverage["first_date"]).dt.days
    coverage["completeness_ratio"] = (
        coverage["row_count"] / (coverage["days_span"] + 1).replace(0, np.nan)
    )
    coverage["completeness_ratio"] = (
        coverage["completeness_ratio"].fillna(0.0).clip(upper=1.0)
    )

    # Row-level flags for extraction metrics
    present_cols = [c for c in EXTRACTION_COLS if c in df.columns]
    if present_cols:
        df["_has_extraction"] = ~df[present_cols].isna().all(axis=1)
    else:
        df["_has_extraction"] = False

    has_resl_min = ("Resl_Min_Price" in df.columns) and df["Resl_Min_Price"].notna()
    has_resl_max = ("Resl_Max_Price" in df.columns) and df["Resl_Max_Price"].notna()
    if isinstance(has_resl_min, pd.Series) and isinstance(has_resl_max, pd.Series):
        df["_has_resale_values"] = has_resl_min & has_resl_max
    else:
        df["_has_resale_values"] = False

    g = df.groupby("EventID")
    extras = g.agg(
        extraction_count=("_has_extraction", "sum"),
        resale_value_count=("_has_resale_values", "sum"),
    ).reset_index()

    coverage = coverage.merge(extras, on="EventID", how="left")

    coverage["extraction_pct"] = (
        coverage["extraction_count"]
        / coverage["row_count"].replace(0, np.nan)
    ).fillna(0.0)

    coverage["completeness_ratio"] = coverage["completeness_ratio"].round(3)
    coverage["extraction_pct"] = coverage["extraction_pct"].round(3)

    coverage = coverage[[
        "EventID", "row_count", "first_date", "last_date", "days_span",
        "completeness_ratio", "extraction_count", "extraction_pct",
        "resale_value_count"
    ]]
    return coverage

# ---------------------------------------------------------------------------
# Resale variability (from Prices.csv)
# ---------------------------------------------------------------------------

def compute_resale_variability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variability = std dev of resale midpoint per EventID:
        midpoint_t = mean([Resl_Min_Price_t, Resl_Max_Price_t], skipna=True)
    Returns: EventID, resale_variability, resale_variability_tiebreak, n_resale_rows
    """
    have_resl_cols = ("Resl_Min_Price" in df.columns) or ("Resl_Max_Price" in df.columns)

    if not have_resl_cols:
        return (
            df[["EventID"]]
            .drop_duplicates()
            .assign(
                resale_variability=0.0,
                resale_variability_tiebreak=0.0,
                n_resale_rows=0
            )
        )

    def _per_event(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("Timestamp").copy()
        any_resale = g[["Resl_Min_Price", "Resl_Max_Price"]].notna().any(axis=1)
        g = g.loc[any_resale]
        if g.empty:
            return pd.Series({
                "resale_variability": 0.0,
                "resale_variability_tiebreak": 0.0,
                "n_resale_rows": 0
            })

        midpoint = g[["Resl_Min_Price", "Resl_Max_Price"]].mean(axis=1, skipna=True)
        resale_std = float(midpoint.std(ddof=1)) if len(midpoint) >= 2 else 0.0

        min_seen = np.nanmin(g[["Resl_Min_Price", "Resl_Max_Price"]].values)
        max_seen = np.nanmax(g[["Resl_Min_Price", "Resl_Max_Price"]].values)
        resale_range = (
            float(max_seen - min_seen)
            if np.isfinite(min_seen) and np.isfinite(max_seen)
            else 0.0
        )

        return pd.Series({
            "resale_variability": resale_std,
            "resale_variability_tiebreak": resale_range,
            "n_resale_rows": int(any_resale.sum()),
        })

    res = (
        df.groupby("EventID", group_keys=False)
        .apply(lambda g: _per_event(g.drop(columns=["EventID"], errors="ignore")))
        .reset_index()
    )

    for col in ["resale_variability", "resale_variability_tiebreak", "n_resale_rows"]:
        if col not in res.columns:
            res[col] = 0.0 if col != "n_resale_rows" else 0

    return res[[
        "EventID",
        "resale_variability",
        "resale_variability_tiebreak",
        "n_resale_rows"
    ]]

# ---------------------------------------------------------------------------
# Resale min / max from Prices.csv
# ---------------------------------------------------------------------------

def compute_resale_min_max(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute min/max resale prices from Prices.csv per EventID:
      - resale_min_price: min(Resl_Min_Price) if present
      - resale_max_price: max(Resl_Max_Price) if present
    """
    agg_map = {}
    if "Resl_Min_Price" in df.columns:
        agg_map["resale_min_price"] = ("Resl_Min_Price", "min")
    if "Resl_Max_Price" in df.columns:
        agg_map["resale_max_price"] = ("Resl_Max_Price", "max")

    if not agg_map:
        return pd.DataFrame(columns=["EventID", "resale_min_price", "resale_max_price"])

    out = df.groupby("EventID").agg(**agg_map).reset_index()

    if "resale_min_price" not in out.columns:
        out["resale_min_price"] = np.nan
    if "resale_max_price" not in out.columns:
        out["resale_max_price"] = np.nan

    return out[["EventID", "resale_min_price", "resale_max_price"]]

# ---------------------------------------------------------------------------
# Ticket-level stats from DB (Tickets + TicketPrices)
# ---------------------------------------------------------------------------

def compute_ticket_stats(event_ids: list[str]) -> pd.DataFrame:
    """
    For each EventID (via Tickets + TicketPrices):
      - ticket_count:        # distinct TicketID in Tickets for that event
      - ticketprice_count:   # rows in TicketPrices for those tickets
      - ticket_min_price:    min TicketPrice for those tickets
      - ticket_max_price:    max TicketPrice for those tickets
    """
    if not event_ids:
        return pd.DataFrame(columns=[
            "EventID",
            "ticket_count",
            "ticketprice_count",
            "ticket_min_price",
            "ticket_max_price",
        ])

    db = DB_API()
    try:
        conn = db.conn
        placeholders = ",".join(["?"] * len(event_ids))

        # One query: Tickets joined to TicketPrices by TicketID, grouped by EventID
        sql = f"""
            SELECT
                t.EventID                                      AS EventID,
                COUNT(DISTINCT t.TicketID)                     AS ticket_count,
                COUNT(tp.TicketPriceRowID)                     AS ticketprice_count,
                MIN(tp.TicketPrice)                            AS ticket_min_price,
                MAX(tp.TicketPrice)                            AS ticket_max_price
            FROM Tickets t
            LEFT JOIN TicketPrices tp
                   ON t.TicketID = tp.TicketID
            WHERE t.EventID IN ({placeholders})
            GROUP BY t.EventID;
        """

        df = pd.read_sql_query(sql, conn, params=event_ids)

        # Ensure all expected columns exist & types are clean
        if "ticket_count" not in df.columns:
            df["ticket_count"] = 0
        if "ticketprice_count" not in df.columns:
            df["ticketprice_count"] = 0
        if "ticket_min_price" not in df.columns:
            df["ticket_min_price"] = np.nan
        if "ticket_max_price" not in df.columns:
            df["ticket_max_price"] = np.nan

        df["ticket_count"] = df["ticket_count"].fillna(0).astype(int)
        df["ticketprice_count"] = df["ticketprice_count"].fillna(0).astype(int)

        return df[[
            "EventID",
            "ticket_count",
            "ticketprice_count",
            "ticket_min_price",
            "ticket_max_price",
        ]]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Event name lookup via DB
# ---------------------------------------------------------------------------

def event_name_lookup(event_id: str) -> str:
    db = DB_API()
    try:
        name = db.get_event_name(event_id)
    finally:
        db.close()
    return name or event_id

# ---------------------------------------------------------------------------
# Top-20 selection
# ---------------------------------------------------------------------------

def select_top_20(coverage: pd.DataFrame, resale_var: pd.DataFrame) -> list[str]:
    tmp = coverage.merge(resale_var, on="EventID", how="left")

    for col in ["resale_variability", "resale_variability_tiebreak", "n_resale_rows"]:
        if col not in tmp.columns:
            tmp[col] = 0

    tmp[["resale_variability", "resale_variability_tiebreak"]] = (
        tmp[["resale_variability", "resale_variability_tiebreak"]].fillna(0.0)
    )
    tmp["n_resale_rows"] = tmp["n_resale_rows"].fillna(0).astype(int)

    order = []
    order += tmp.sort_values("row_count", ascending=False)["EventID"].tolist()
    order += tmp.sort_values("extraction_pct", ascending=False)["EventID"].tolist()
    order += tmp.sort_values("resale_value_count", ascending=False)["EventID"].tolist()
    order += tmp.sort_values(
        ["resale_variability", "resale_variability_tiebreak"],
        ascending=False
    )["EventID"].tolist()

    seen, top = set(), []
    for eid in order:
        if eid not in seen:
            top.append(eid)
            seen.add(eid)
        if len(top) == 20:
            break

    if len(top) < 20:
        for eid in tmp.sort_values("row_count", ascending=False)["EventID"]:
            if eid not in seen:
                top.append(eid)
                seen.add(eid)
            if len(top) == 20:
                break

    return top

def select_top_by_volatility(resale_var: pd.DataFrame, k: int = 10) -> list[str]:
    if resale_var.empty:
        return []
    tmp = resale_var.copy()
    tmp["resale_variability"] = tmp["resale_variability"].fillna(0.0)
    tmp["resale_variability_tiebreak"] = tmp["resale_variability_tiebreak"].fillna(0.0)
    tmp = tmp.sort_values(
        ["resale_variability", "resale_variability_tiebreak"],
        ascending=False
    )
    return tmp["EventID"].head(k).tolist()

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_standardized_three_lines(
    df: pd.DataFrame,
    event_ids: list[str],
    outdir: Path,
):
    """
    For each EventID, plot standardized (0–1) lines:
      - Resl_Min_Price
      - Resl_Max_Price
      - Resl_Count
    """
    plots_dir = outdir / "top20_standardized_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    have_cols = {
        "min": "Resl_Min_Price" in df.columns,
        "max": "Resl_Max_Price" in df.columns,
        "cnt": "Resl_Count" in df.columns,
    }
    if not any(have_cols.values()):
        print("No Resl_* columns present to plot.")
        return

    for eid in event_ids:
        g = df[df["EventID"] == eid].sort_values("Timestamp").copy()
        if g.empty:
            continue

        lines = []
        labels = []

        if have_cols["min"]:
            y_min = minmax01(g["Resl_Min_Price"])
            if y_min.notna().any():
                lines.append((g["Timestamp"], y_min))
                labels.append("Resl_Min_Price (std)")

        if have_cols["max"]:
            y_max = minmax01(g["Resl_Max_Price"])
            if y_max.notna().any():
                lines.append((g["Timestamp"], y_max))
                labels.append("Resl_Max_Price (std)")

        if have_cols["cnt"]:
            y_cnt = minmax01(g["Resl_Count"])
            if y_cnt.notna().any():
                lines.append((g["Timestamp"], y_cnt))
                labels.append("Resl_Count (std)")

        if not lines:
            continue

        title = f"{event_name_lookup(eid)} (EventID: {eid}) — Standardized 0–1"

        fig, ax = plt.subplots()
        for (x, y), lbl in zip(lines, labels):
            ax.plot(x, y, label=lbl)

        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Standardized Value (0–1)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{eid}_std.png")
        plt.close(fig)

def plot_resale_min_max(
    df: pd.DataFrame,
    event_ids: list[str],
    outdir: Path,
    overlay_counts_for: set[str] | None = None
):
    """
    Plot Resl_Min_Price / Resl_Max_Price for given events.
    Optionally overlay Resl_Count on a secondary axis for selected events.
    """
    plots_dir = outdir / "top20_resale_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    overlay_counts_for = overlay_counts_for or set()

    for eid in event_ids:
        g = df[df["EventID"] == eid].sort_values("Timestamp")
        if g.empty:
            continue
        if ("Resl_Min_Price" not in g.columns) and ("Resl_Max_Price" not in g.columns):
            continue

        title = f"{event_name_lookup(eid)} (EventID: {eid})"

        fig, ax = plt.subplots()
        if "Resl_Min_Price" in g.columns:
            ax.plot(g["Timestamp"], g["Resl_Min_Price"], label="Resl_Min_Price")
        if "Resl_Max_Price" in g.columns:
            ax.plot(g["Timestamp"], g["Resl_Max_Price"], label="Resl_Max_Price")

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        # Secondary axis for Resl_Count if requested
        if (eid in overlay_counts_for) and ("Resl_Count" in g.columns) and g["Resl_Count"].notna().any():
            ax2 = ax.twinx()
            ax2.plot(g["Timestamp"], g["Resl_Count"], label="Resl_Count", linestyle="--")
            ax2.set_ylabel("Resale Listings (Resl_Count)")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(plots_dir / f"{eid}_resale_min_max.png")
        plt.close(fig)

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    outdir = Path("./analysis_out")
    outdir.mkdir(parents=True, exist_ok=True)

    # Adjust this path if needed
    prices_csv = "/Users/connorkeefe/PycharmProjects/TicketMaster/Scripts/Prices.csv"

    # 1) Load & clean Prices.csv
    df = read_and_clean(prices_csv)

    # 2) Coverage metrics
    coverage = compute_event_coverage(df)

    # 3) Resale variability (Prices.csv)
    resale_var = compute_resale_variability(df)

    # 4) Resale min/max (Prices.csv)
    resale_minmax = compute_resale_min_max(df)

    # 5) Ticket stats (Tickets + TicketPrices via DB)
    event_ids = coverage["EventID"].tolist()
    ticket_stats = compute_ticket_stats(event_ids)

    # 6) Combine everything into event_coverage.csv
    coverage_out = (
        coverage
        .merge(resale_var, on="EventID", how="left")
        .merge(resale_minmax, on="EventID", how="left")
        .merge(ticket_stats, on="EventID", how="left")
    )

    coverage_out.to_csv(outdir / "event_coverage.csv", index=False)

    # 7) Top-20 selection
    top20_ids = select_top_20(coverage, resale_var)

    # 8) Top-K by volatility for optional overlays
    top_vol_ids = set(select_top_by_volatility(resale_var, k=10))

    # 9) Plots
    plot_standardized_three_lines(df, top20_ids, outdir)
    # If you want resale plots with overlay, uncomment:
    # plot_resale_min_max(df, top20_ids, outdir, overlay_counts_for=top_vol_ids)

    # 10) Quick preview
    print("=== Coverage + Variability + Ticket & Resale Stats (head) ===")
    print(coverage_out.head(10))
    print(f"\nTop 20 EventIDs selected ({len(top20_ids)}):")
    print(top20_ids)
    print(f"\nTop volatility EventIDs (for optional overlay) ({len(top_vol_ids)}):")
    print(list(top_vol_ids))
    print("\nOutputs written to:", outdir.resolve())

if __name__ == "__main__":
    main()

