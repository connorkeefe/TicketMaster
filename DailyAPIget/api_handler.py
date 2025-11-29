import requests
import uuid
from datetime import datetime, timedelta
from run_flags import TEST as TEST_FLAG
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger


API_KEY = os.getenv("TM_API_KEY")
# Test Toggle to limit the return of API call

DAY_RANGE = 90
PAGE_SIZE = 1 if TEST_FLAG else 150
MAX_PAGE = 1 if TEST_FLAG else 5 #Limited by 1000 (200 * 5)


def list_events(c_code, segments):
    # Define the endpoint and parameters
    dates = create_date_ranges(DAY_RANGE)
    event_list = []
    for segment in segments:
        for (start, end) in dates:
            elist, pages = api_event_list(c_code, start, end, 0, segment)
            event_list.extend(elist)
            pages = min(pages, MAX_PAGE)
            for i in range(1, pages):
                elist, _ = api_event_list(c_code, start, end, i, segment)
                event_list.extend(elist)
        
    logger.info(f"{len(set(event_list))} unique event ids generated")

    return list(set(event_list))

def api_event_list(c_code, start, end, page, segment = None):
    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    if segment == "Music":
        params = {
        "apikey": API_KEY,
        "countryCode": c_code,
        "startDateTime": start,
        "endDateTime": end,
        "size": 1 if TEST_FLAG else 75,
        "page": page,
        "segmentName": segment
        }
    elif segment:
        params = {
        "apikey": API_KEY,
        "countryCode": c_code,
        "startDateTime": start,
        "endDateTime": end,
        "size": PAGE_SIZE,
        "page": page,
        "segmentName": segment
        }
    else:
        params = {
            "apikey": API_KEY,
            "countryCode": c_code,
            "startDateTime": start,
            "endDateTime": end,
            "size": PAGE_SIZE,
            "page": page
        }

    # Make the GET request
    response = requests.get(url, params=params)

    data = response.json()
    events = data.get("_embedded", {}).get("events", [])
    pages = data.get("page", {}).get("totalPages", None)
    event_list = []
    for event in events:
        id = event.get("id", None)
        if id:
            event_list.append(id)

    return event_list, pages

def get_event_details(id, nhl_df, nfl_df, nba_df, artist_df):
    data = get_event_api(id)

    venue, venue_id = generate_venue(data)
    attraction1id, name1, attraction2id, name2 = generate_attractions(data)
    attraction = create_attraction(attraction1id, name1, attraction2id, name2)
    event, segment, subGenre = generate_event(id, venue_id, attraction1id, attraction2id, data)
    df = None
    if segment == 'Sports':
        if subGenre == 'NBA':
            df = nba_df
        elif subGenre == 'NHL':
            df = nhl_df
        elif subGenre == 'NFL':
            df = nfl_df
    elif segment == 'Music':
        df = artist_df
    attraction_instance, attraction_instance_id = generate_attraction_instance(attraction1id, name1, attraction2id, name2, df)
    price = generate_price(id, attraction_instance_id, data)

    return {"venue": venue, "attraction": attraction, "event": event, "attraction_instance": attraction_instance, "price": price}

def get_event_api(id):
    url = f"https://app.ticketmaster.com/discovery/v2/events/{id}.json"
    params = {
        "apikey": API_KEY,
    }

    # Make the GET request
    response = requests.get(url, params=params)
    data = response.json()

    return data

# This API requires partner level API key (difficult to get)
def get_inventory_api(id):
    url = f"https://app.ticketmaster.com/inventory-status/v1/availability.json"
    params = {
        "apikey": API_KEY,
        "events": id
    }

    # Make the GET request
    response = requests.get(url, params=params)
    data = response.json()

    return data

def generate_venue(data):
    venue = []

    venue_info = data.get("_embedded", {}).get("venues", [{}])[0]
    venue_id = venue_info.get("id", None)
    if venue_id:
        venue.append(venue_id)
        venue.append(venue_info.get("name", None))
        venue.append(venue_info.get("city", {}).get("name", None))
        venue.append(venue_info.get("state", {}).get("stateCode", None))
        venue.append(venue_info.get("country", {}).get("countryCode", None))

    return tuple(venue), venue_id

def generate_attractions(data):
    attraction_info = data.get("_embedded", {}).get("attractions", [{}])
    name1 = None
    name2 = None
    attraction1id = None
    attraction2id = None
    if attraction_info[0]:
        attraction1id = attraction_info[0].get("id", None)
        name1 = attraction_info[0].get("name", None)
    if len(attraction_info) > 1 and attraction_info[1]:
        attraction2id = attraction_info[1].get("id", None)
        name2 = attraction_info[1].get("name", None)
    return attraction1id, name1, attraction2id, name2

def create_attraction(attraction1id, name1, attraction2id, name2):
    attraction = []
    if attraction1id:
        attraction.append((attraction1id, name1))
    if attraction2id:
        attraction.append((attraction2id, name2))
    return attraction


def generate_event(id, venue_id, attraction1id, attraction2id, data):
    event = []

    event.append(id)
    event.append(venue_id)
    event.append(data.get("url", None))
    event.append(attraction1id)
    event.append(attraction2id)
    sales_info = data.get("sales", {}).get("public", {})
    event.append(create_date(sales_info.get("startDateTime", None)))
    event.append(create_date(sales_info.get("endDateTime", None)))
    event.append(data.get("name", None))
    date_info = data.get("dates", {})
    event.append(create_date(date_info.get("start", {}).get("dateTime", None)))
    event.append(date_info.get("timezone", None))
    classification = data.get("classifications", [{}])[0]
    segment = classification.get("segment", {}).get("name", None)
    event.append(segment)
    event.append(classification.get("genre", {}).get("name", None))
    subGenre = classification.get("subGenre", {}).get("name", None)
    event.append(subGenre)
    event.append(classification.get("type", {}).get("name", None))
    event.append(classification.get("subType", {}).get("name", None))

    return tuple(event), segment, subGenre

def generate_attraction_instance(attraction1id, name1, attraction2id, name2, df):
    attraction_instance = []
    attraction_instance_id = generate_timestamped_uuid()
    attraction_instance.append(attraction_instance_id)
    attraction_instance.append(attraction1id)
    result1 = None
    result2 = None
    if df is not None:
        if name1:
            result1 = df.loc[df['Name'] == name1, 'Rank'].iloc[0] if name1 in df['Name'].values else None
            if result1:
                result1 = int(result1)
        if name2:
            result2 = df.loc[df['Name'] == name2, 'Rank'].iloc[0] if name2 in df['Name'].values else None
            if result2:
                result2 = int(result2)
    attraction_instance.append(result1)
    attraction_instance.append(attraction2id)
    attraction_instance.append(result2)

    return tuple(attraction_instance), attraction_instance_id


def generate_price(id, attraction_instance_id, data):
    price = []
    price.append(generate_timestamped_uuid())
    price.append(id)
    price.append(attraction_instance_id)
    price_info = data.get("priceRanges", [{}])[0]
    price.append(price_info.get("min", None))
    price.append(price_info.get("max", None))
    price.append(price_info.get("currency", None))
    return tuple(price)
    

def create_date(date_str):
    if date_str:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    else:
        return date_str

def generate_timestamped_uuid():
    # Get the current timestamp in ISO format
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    
    # Generate a random UUID
    random_uuid = uuid.uuid4()
    
    # Combine the timestamp with the UUID
    timestamped_uuid = f"{timestamp}-{random_uuid}"
    
    return timestamped_uuid

def create_date_ranges(days: int):
    # Get the current date and the end date after the specified number of months
    start_date = datetime.utcnow()
    end_date = start_date + timedelta(days=days)
    
    # Calculate the length of each range in days
    total_days = (end_date - start_date).days
    range_length = total_days // 4  # Divide into 4 equal ranges
    
    # Generate the date ranges
    date_ranges = []
    current_start = start_date
    for _ in range(4):
        current_end = current_start + timedelta(days=range_length)
        # Make sure the last range ends exactly on the end date
        if current_end > end_date:
            current_end = end_date
        date_ranges.append((current_start.strftime('%Y-%m-%dT%H:%M:%SZ'), current_end.strftime('%Y-%m-%dT%H:%M:%SZ')))
        current_start = current_end + timedelta(days=1)  # Start next range after current end
    
    return date_ranges



