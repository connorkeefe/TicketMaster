from DailyAPIget import lambda_function
from DailyPriceScraper import selenium_runner
from DailyDataPull import pull_training_data
from Inventory_API import scrape_inventory
from logger import logger
import datetime


if __name__ == "__main__":
    current_day = datetime.datetime.now().day
    # Check if the day is even or odd
    if current_day % 2 == 0:
        event = {'c_code': 'CA'}
    else:
        event = {'c_code': 'US'}
    logger.info(f"lambda payload: {event}")
    logger.info("Starting Ticketmaster API Call")
    lambda_function.lambda_handler(event, None)
    logger.info("Finished Ticketmaster API Call, Starting Selenium Price Scrape")
    selenium_runner.run_selenium()
    # scrape_inventory.run_inventory_api()
    logger.info("Finished Selenium Price Scrape")
    # logger.info("Running Train Pull")
    # pull_training_data.pull_train_data()