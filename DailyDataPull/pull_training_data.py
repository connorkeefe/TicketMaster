import json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db_api import DB_API

TRAIN_DATA_QUERY = """
WITH PriceFeatures AS (
    SELECT
        p.Timestamp,
        p.EventID,
        p.AttractionInstanceID,
        p.Stan_Min_Price,
        p.Stan_Max_Price,
        p.Resl_Min_Price,
        p.Resl_Max_Price
    FROM Prices p
),
EventFeatures AS (
    SELECT
        e.EventID,
        e.VenueID,
        e.sale_start,
        e.sale_end,
        e.event_start,
        e.name AS Event_name,
        e.AttractionID1,
        e.AttractionID2,
        e.segment,
        e.genre,
        e.subgenre,
        v.name AS Venue_Name,
        v.city,
        v.state,
        v.country
    FROM Events e
    JOIN Venues v ON e.VenueID = v.VenueID
),
AttractionFeatures AS (
    SELECT
        a1.AttractionID AS Attraction1ID,
        a1.name AS Attraction1_name,
        a2.AttractionID AS Attraction2ID,
        a2.name AS Attraction2_name,
        aid.AttractionInstanceID,
        aid.Attraction1Rank,
        aid.Attraction2Rank
    FROM AttractionInstanceDetail aid
    JOIN Attractions a1 ON aid.AttractionID1 = a1.AttractionID
    LEFT JOIN Attractions a2 ON aid.AttractionID2 = a2.AttractionID
),
CombinedData AS (
    SELECT
        ROW_NUMBER() OVER (PARTITION BY pf.EventID ORDER BY TO_TIMESTAMP(SUBSTRING(pf.Timestamp, 1, 8), 'YYYYMMDDHH24MISS')) AS time_idx,
        pf.Timestamp,
        pf.EventID,
        pf.AttractionInstanceID,
        ef.VenueID,
        ef.sale_start,
        ef.sale_end,
        ef.event_start,
        af.Attraction1Rank,
        af.Attraction2Rank,
        pf.Stan_Min_Price,
        pf.Stan_Max_Price,
        pf.Resl_Min_Price,
        pf.Resl_Max_Price,
        ef.Event_name,
        af.Attraction1_name,
        af.Attraction2_name,
        ef.city,
        ef.state,
        ef.country,
        ef.segment,
        ef.genre,
        ef.subgenre,
        ef.Venue_Name
    FROM PriceFeatures pf
    JOIN EventFeatures ef ON pf.EventID = ef.EventID
    JOIN AttractionFeatures af ON pf.AttractionInstanceID = af.AttractionInstanceID
    WHERE ef.segment= 'Sports' AND ef.country = 'US' AND ef.subgenre = 'NBA'
)
SELECT *
FROM CombinedData
LIMIT 1000;
"""


def pull_train_data():
    db = DB_API()

    df = db.get_train_data(TRAIN_DATA_QUERY)
    df.to_csv(r"csvs\train_test_sports.csv", index=False)

    db.close()


