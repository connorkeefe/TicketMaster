from DailyAPIget import get_events
from Inventory_API import scrape_inventory
from DailyMLRun import ml_daily_run
from run_flags import RUN, ML_RUN
from db.db_api import insert_and_backup
from logger import logger

if __name__ == "__main__":
    event = {'c_code': 'US'}
    logger.info(f"lambda payload: {event}")
    logger.info("Starting Ticketmaster API Call")
    get_events.fetch_and_save_events(event, None)
    logger.info("Finished Ticketmaster API Call")
    if RUN:
        logger.info("Starting Inventory API Price Scrape")
        scrape_inventory.run_inventory_api()
        logger.info("Finished Inventory API Price Scrape")
        if not ML_RUN:
            insert_and_backup()
    else:
        logger.info("Finished db operation successfully")

    if ML_RUN:
        logger.info("Starting ML daily run")
        ml_daily_run.ml_daily_run()
        logger.info("Finished ML DailY RUN")
