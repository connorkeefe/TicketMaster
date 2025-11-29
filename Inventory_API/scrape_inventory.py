import json
import random
from db.db_api import DB_API
from run_flags import SCRAPE_TICKETS
from Inventory_API.camoufox_runner import StealthWeb
from Inventory_API.parse_utils import EVENTID, URL, parse_facet, find_event_results, parse_offers_to_rows
from Inventory_API.requests_runner import PROXY, get_response_requests, get_ticket_response_requests, get_response
from Inventory_API.section_calculation import populate_section_market_data
from logger import logger
import pandas as pd
import time
import tqdm

def get_price_id_df():
    db = DB_API()
    df = db.get_event_ids_prices()
    logger.info(df.head())
    logger.info(len(df))
    db.close()
    return df

def update_sections_and_insert():
    db = DB_API()
    db.populate_section_table()
    db.populate_section_prices()
    populate_section_market_data(db)
    db.close()

def inventory_api(df):
    df['Stan_Min'] = None
    df['Stan_Max'] = None
    df['Resl_Min'] = None
    df['Resl_Max'] = None
    df['Stan_Count'] = None
    df['Resl_Count'] = None
    error_df = pd.DataFrame(columns=df.columns)
    ticket_rows = []
    request_headers = None
    count = 0
    error_count = 0
    ticket_count = 0
    ticket_error_count = 0
    start_time = time.time()
    url_cookie = 'https://www.ticketmaster.com/capitals-vs-maple-leafs-washington-district-of-columbia-11-28-2025/event/150062F0F076601F'
    for index, event in tqdm.tqdm(df.iterrows(), total=len(df)):
        timestamp_uuid = event.get("Timestamp", None)
        id = event.get(EVENTID, None)
        process, response, data, heads, process_ticket, ticket_response, ticket_body = get_response(event)
        if process:
            logger.info(f"Running Camoufox Block: {id}")
            request_headers = heads
            logger.info(f"Response: {response}, {id}")
            # logger.info(f"Data: {data}, {id}")
            facets = data.get("facets", [])
            # logger.info(f"Facets: {facets}, {id}")
            if len(facets) > 0:
                stan_min, stan_max, resl_min, resl_max, stan_count, resl_count = parse_facet(facets)
                logger.info(f"Results: {stan_min}, {stan_max}, {resl_min}, {resl_max}, {stan_count}, {resl_count}")
                df.at[index, 'Stan_Min'] = stan_min
                df.at[index, 'Stan_Max'] = stan_max
                df.at[index, 'Resl_Min'] = resl_min
                df.at[index, 'Resl_Max'] = resl_max
                df.at[index, 'Stan_Count'] = stan_count
                df.at[index, 'Resl_Count'] = resl_count
            else:
                logger.info(f"No Results, {id}")

        elif id:
            logger.info(f"Running Request Block, {id}")
            if not request_headers:
                stealth_web = StealthWeb(url_cookie, proxy=PROXY)
                stealth_web.get_headers()
                headers = stealth_web.headers
            else:
                headers = request_headers
            logger.info(f"Got Headers: {headers}, {id}")
            process, response, data = get_response_requests(event, headers)
            if process and response.status_code == 200:
                logger.info(f"Response: {response}, {id}")
                # logger.info(f"Data: {data}, {id}")
                facets = data.get("facets", [])
                # logger.info(f"Facets: {facets}, {id}")
                if len(facets) > 0:
                    stan_min, stan_max, resl_min, resl_max, stan_count, resl_count = parse_facet(facets)
                    logger.info(f"Results: {stan_min}, {stan_max}, {resl_min}, {resl_max}, {stan_count}, {resl_count}")
                    df.at[index, 'Stan_Min'] = stan_min
                    df.at[index, 'Stan_Max'] = stan_max
                    df.at[index, 'Resl_Min'] = resl_min
                    df.at[index, 'Resl_Max'] = resl_max
                    df.at[index, 'Stan_Count'] = stan_count
                    df.at[index, 'Resl_Count'] = resl_count
                else:
                    logger.info(f"No Results, {id}")
            else:
                logger.info(f"Failed Request Block, {response.status_code if response else None}, {id}")
                error_count += 1
        else:
            logger.info(f"No Block Executed: {response}, {event.get(URL, None)}, {id}")
            error_count += 1
        count += 1
        if count % 10 == 0:
            logger.info(
                f"10 complete, error rate: {error_count / count}")

        if SCRAPE_TICKETS:
            # process, response, data, heads = get_ticket_response(event)
            response, data = ticket_response, ticket_body
            if process_ticket:
                logger.info(f"Running Ticket Camoufox Block: {id}")
                request_headers = heads
                # logger.info(f"Response: {response}, {id}")
                # logger.info(f"Data: {data}, {id}")
                tickets = data.get("_embedded", {}).get("offer", [])
                logger.info(f"Length Result: {len(tickets)}, {id}")
                if len(tickets) > 0:
                    ticket_rows_add = parse_offers_to_rows(timestamp_uuid, id, tickets)
                    if ticket_rows_add:
                        ticket_rows.extend(ticket_rows_add)

                else:
                    logger.info(f"No Results, {id}")

            elif id:
                logger.info(f"Running Ticket Request Block, {id}")
                if not request_headers:
                    stealth_web = StealthWeb(url_cookie, proxy=PROXY)
                    stealth_web.get_headers()
                    headers = stealth_web.headers
                else:
                    headers = request_headers
                logger.info(f"Got Headers: {headers}, {id}")
                process, response, data = get_ticket_response_requests(event, headers)
                if process and response.status_code == 200:
                    logger.info(f"Response: {response}, {id}")
                    # logger.info(f"Data: {data}, {id}")
                    tickets = data.get("_embedded", {}).get("offer", [])
                    logger.info(f"Length Result: {len(tickets)}, {id}")
                    if len(tickets) > 0:
                        ticket_rows_add = parse_offers_to_rows(timestamp_uuid, id, tickets)
                        if ticket_rows_add:
                            ticket_rows.extend(ticket_rows_add)
                    else:
                        logger.info(f"No Results, {id}")
                else:
                    logger.info(f"Failed Ticket Request Block, {response.status_code if response else None}, {id}")
                    ticket_error_count += 1
                    if response and response.status_code == 403:
                        logger.info(f"Inserting {id} into rerun_df")
                        error_df.loc[len(error_df)] = event


            else:
                logger.info(f"No Ticket Block Executed: {response}, {event.get(URL, None)}, {id}")
                ticket_error_count += 1
            ticket_count += 1
            if ticket_count % 10 == 0:
                logger.info(
                    f"10 Event Tickets complete, error rate: {error_count / count}")
            max_elements = 500000
            if len(ticket_rows) > max_elements:
                logger.info(f"Ticket rows exceeded max elements of {max_elements}, performing insert")
                db = DB_API()
                db.upsert_tickets_and_prices(ticket_rows)
                ticket_rows = []
                db.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(df.head())
    if count > 0:
        logger.info(f"Error %: {error_count / count}")
    logger.info(f"Elapsed Minutes: {elapsed_time / 60}")
    db = DB_API()
    db.update_prices(df)
    if SCRAPE_TICKETS:
        db_start_time = time.time()
        db.upsert_tickets_and_prices(ticket_rows)
        db_end_time = time.time()
        logger.info(f"Upsert complete in {db_end_time - db_start_time} seconds")
        logger.info(f"Ticket Error %: {ticket_error_count / ticket_count}")
    db.close()
    return error_df

def run_inventory_api():
    df = get_price_id_df()
    retries = 0
    logger.info("Running First Pass")
    rerun_df = inventory_api(df)
    while len(rerun_df) > 0 and retries < 2:
        logger.info(f"Running Retry Pass #{retries + 1} after waiting 5 minutes, df length = {len(rerun_df)}")
        time.sleep(300)
        rerun_df = inventory_api(rerun_df)
        retries += 1

    logger.info("All Inventory API Runs Complete")
    logger.info("Updating sections and inserting records")
    update_sections_and_insert()


