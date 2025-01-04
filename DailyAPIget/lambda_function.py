import json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db_api import DB_API
from DailyAPIget.web_scraper import DailyScraper
from DailyAPIget import api_handler
# import boto3

# Delete db toggle
DELETE = False
# Create tables toggle
CREATE = False
# Run API toggle
RUN = True
# Test mode toggle
TEST = False

def lambda_handler(event, context):
    if RUN:
        c_code = event.get('c_code', None)
        nhl_df, nfl_df, nba_df, artist_df = scraper_process()
        events, venues, prices, attractions, attraction_instances = api_process(c_code, nhl_df, nfl_df, nba_df, artist_df)
        db_process(events, venues, prices, attractions, attraction_instances)
    elif TEST:
        # data = api_test(event)
        df = collect_data()
        # data = None
    else:
        events, venues, prices, attractions, attraction_instances = 0,0,0,0,0
        db_process(events, venues, prices, attractions, attraction_instances)

    if TEST:
        return {
        'statusCode': 200,
        'body': json.dumps(data)
        }

    return {
        'statusCode': 200,
        'body': "Successfull Execution"
    }


def api_process(c_code, nhl_df, nfl_df, nba_df, artist_df):
    event_list = api_handler.list_events(c_code)

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
        
    db.close()

def scraper_process():
    scraper = DailyScraper()
    scraper.run_daily_scrape()
    nhl_df = scraper.get_table("NHL")
    nfl_df = scraper.get_table("NFL")
    nba_df = scraper.get_table("NBA")
    artist_df = scraper.get_table("Artist100")

    return nhl_df, nfl_df, nba_df, artist_df

def collect_data():
    db = DB_API()
    df = db.fetch_event_prices()
    filtered_ids = df['eventid'].value_counts()
    df['id_count'] = df['eventid'].map(filtered_ids)
    bucket = 'ticket-generator-storage'
    key = 'price_data.csv'
    # save_to_s3(df, bucket, key)


def api_test(event):
    c_code = event.get('c_code')
    ranges = api_handler.create_date_ranges(180)
    (start, end) = ranges[0]
    event_list, _ = api_handler.api_event_list(c_code, start, end, 0, 'Music')
    id_test = event_list[0]
    data = api_handler.get_event_api(id_test)
    # url = data.get("url")
    # data = None
    # url = "https://example.com"
    # scraper = Selenium_Scraper()
    # content = scraper.get_url(url)
    # scraper.close()
    # # logger.info(content)
    # bucket = 'ticket-generator-storage'
    # key = 'price_site.txt'
    # save_soup_to_s3(content, bucket, key)
    return data

# def save_to_s3(df, bucket, key):
#     # Convert the DataFrame to CSV format in-memory
#     csv_buffer = io.StringIO()
#     df.to_csv(csv_buffer, index=False)
#
#     # Initialize S3 client
#     s3_client = boto3.client('s3')
#
#     # Upload the CSV file
#     s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
#     logger.info(f"DataFrame saved to s3://{bucket}/{key}")

# def save_soup_to_s3(soup, bucket, key):
#     # page_text = soup.prettify()  # Get all text from the page
#
#     # Use StringIO to create an in-memory file-like object for text
#     file_obj = io.StringIO()
#     file_obj.write(soup)
#     file_obj.seek(0)  # Move the pointer to the start of the file
#     s3_client = boto3.client('s3')
#     # Upload to S3
#     s3_client.put_object(Bucket=bucket, Key=key, Body=file_obj.getvalue(), ContentType='text/plain')

