import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db_api import DB_API
from db.db_api_ml import DB_API_ML
from DailyAPIget.web_scraper import DailyScraper
from DailyAPIget import api_handler
from Inventory_API.section_calculation import populate_section_market_data
from run_flags import DELETE, CREATE, RUN, BACKUP, TEST


def fetch_and_save_events(event, context):
    if RUN:
        run = input(
            f"RUN: Delete flag is: {DELETE}, Create flag is {CREATE}, Test flag is {TEST}, are your sure you want to proceed: (yes/no)")
        if run == 'yes':
            c_code = event.get('c_code', None)
            nhl_df, nfl_df, nba_df, artist_df = scraper_process()
            events, venues, prices, attractions, attraction_instances = api_process(c_code, nhl_df, nfl_df, nba_df, artist_df)
            db_process(events, venues, prices, attractions, attraction_instances)
    else:
        delete = input(f"Delete flag is: {DELETE}, Create flag is {CREATE}, Test flag is {TEST}, are your sure you want to proceed: (yes/no)")
        if delete == 'yes':
            events, venues, prices, attractions, attraction_instances = 0,0,0,0,0
            db_process(events, venues, prices, attractions, attraction_instances)
        else:
            print("Cancelled run")


def api_process(c_code, nhl_df, nfl_df, nba_df, artist_df):
    event_list = api_handler.list_events(c_code, ["Sports", "Music"])

    events = []
    venues = []
    prices = []
    attractions = []
    attraction_instances = []
    for id in event_list:
        event_details = api_handler.get_event_details(id, nhl_df, nfl_df, nba_df, artist_df)
        events.append(event_details["event"])
        attractions.extend(event_details["attraction"])
        if len(event_details["venue"]) > 0:
            venues.append(event_details["venue"])
        attraction_instances.append(event_details["attraction_instance"])
        prices.append(event_details["price"])

    return events, venues, prices, attractions, attraction_instances

def db_process(events, venues, prices, attractions, attraction_instances):
    db = DB_API()
    db_ml = DB_API_ML()
    if RUN:
        db.batch_insert_venues(venues)
        db.batch_insert_attractions(attractions)
        db.batch_insert_events(events)
        db.batch_insert_attraction_instances(attraction_instances)
        db.batch_insert_prices(prices)
    if DELETE:
        db.drop_tables()
        # db.drop_attraction_instance()
    if CREATE:
        db.create_tables()

    if BACKUP:
        db.backup_database()

    db.close()

def scraper_process():
    scraper = DailyScraper()
    scraper.run_daily_scrape()
    nhl_df = scraper.get_table("NHL")
    nfl_df = scraper.get_table("NFL")
    nba_df = scraper.get_table("NBA")
    artist_df = scraper.get_table("Artist100")

    return nhl_df, nfl_df, nba_df, artist_df
