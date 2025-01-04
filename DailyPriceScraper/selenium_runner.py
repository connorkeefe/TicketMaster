
import os, sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db_api import DB_API
from logger import logger
import random

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import re
import time

USER_AGENTS = [
    # Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",

    # macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",

    # Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:116.0) Gecko/20100101 Firefox/116.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:114.0) Gecko/20100101 Firefox/114.0",

    # Older Windows Versions
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",

    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.0.0"
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
]

DRIVER_PATH = r"/Users/connorkeefe/PycharmProjects/TicketMaster/DailyPriceScraper/driver/chromedriver"

# WINDOW_SIZES = [
#     # "1920x1080",
#     # "1366x768",
#     # "1440x900",
#     # "1600x900",
#     "1280x800"
# ]

# Proxy Config
USERNAME = ""
PASSWORD = ""
ENDPOINT = ""


def chrome_proxy(user: str, password: str, endpoint: str) -> dict:
    wire_options = {
        "proxy": {
            "http": f"http://{user}:{password}@{endpoint}",
            "https": f"https://{user}:{password}@{endpoint}",
        }
    }

    return wire_options

# In[2]:
def get_price_df():
    db = DB_API()
    df = db.get_prices()
    db.close()
    return df


# In[3]:
def initialize_driver():
    driver_dict = {}
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    # chrome_options.add_argument("--ignore-certificate-errors")
    user_agent = random.choice(USER_AGENTS)
    driver_dict['user_agent'] = user_agent
    chrome_options.add_argument(f"user-agent={user_agent}")
    driver_dict['disable_blink'] = random.choice([False])
    if driver_dict['disable_blink']:
        driver_dict['disable_blink'] = True
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    driver_dict['disable_gpu'] = random.choice([False])
    if driver_dict['disable_gpu']:
        chrome_options.add_argument("--disable-gpu")
    # window_size = random.choice(WINDOW_SIZES)
    # driver_dict['window_size'] = window_size
    # chrome_options.add_argument(f"--window-size={window_size}")
    driver_dict['no-sbx'] = random.choice([False])
    if driver_dict['no-sbx']:
        chrome_options.add_argument("--no-sandbox")
    driver_dict['disable_dev'] = random.choice([True])
    if driver_dict['disable_dev']:
        chrome_options.add_argument("--disable-dev-shm-usage")
    driver_dict['exclude_switches'] = random.choice([True])
    if driver_dict['exclude_switches']:
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver_dict['use_auto_extension'] = random.choice([False])
    if driver_dict['use_auto_extension']:
        chrome_options.add_experimental_option("useAutomationExtension", False)
    # proxies = chrome_proxy(USERNAME, PASSWORD, ENDPOINT)
    driver = webdriver.Chrome(service=Service(DRIVER_PATH), options=chrome_options)
    driver.maximize_window()
    return driver, driver_dict

def reset_driver(driver, driver_list, driver_count, min_wait, max_wait):
    driver.quit()
    prev_driver = driver_list.pop()
    prev_driver['driver_count'] = driver_count
    driver_list.append(prev_driver)
    wait_time = random.uniform(min_wait, max_wait)
    time.sleep(wait_time)
    driver, driver_dict = initialize_driver()
    driver_list.append(driver_dict)
    return driver, driver_list

def strip_non_digit_decimal(input_string):
    return re.sub(r'[^0-9.]+', '', input_string)

def c_avg(numbers):
    return sum(numbers) / len(numbers) if numbers else 0  # Avoid division by zero if list is empty

def cookies_available(driver):
    # Return True if cookies are available
    return len(driver.get_cookies()) > 0

def get_cookies(url):
    driver, _ = initialize_driver()
    driver.get(url)
    time.sleep(5)
    page_source = driver.page_source
    page_size_bytes = len(page_source.encode('utf-8'))
    cookies = driver.get_cookies()
    cookie_header = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies])
    cookie_header_size_bytes = len(cookie_header.encode('utf-8'))
    # Close the browser
    driver.quit()
    logger.info(f"Size of page: {page_size_bytes}")
    logger.info(f"Size of cookies: {cookie_header_size_bytes}")
    return cookie_header

def process(url, driver):
    error = False
    no_best = False
    block_page = False
    wait = WebDriverWait(driver, random.choice([3,4,5,6]))
    low_dict = []
    low_dict_resl = []
    high_dict = []
    high_dict_resl = []
    try:
        driver.get(str(url))
        # logger.debug(driver.page_source)
    except Exception as e:
        error = True
        return c_avg(low_dict), c_avg(low_dict_resl), c_avg(high_dict), c_avg(high_dict_resl), error, no_best, block_page
    try:
        accept_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-bdd="accept-modal-accept-button"]'))
        )
        accept_button.click()
        path = os.path.join("jpgs", f"{str(url).replace('/', '').replace(':','')}.png")
        # logger.debug(f"screenshot: {path}")
        driver.save_screenshot(path)
    except Exception as e:
        path = os.path.join("jpgs", f"{str(url).replace('/', '').replace(':', '')}.png")
        # logger.debug(f"screenshot: {path}")
        driver.save_screenshot(path)
        # logger.info("Error: Button not found or could not be clicked.", e)
        error = False
        # Find all the ticket items in the quick-picks list

    try:
        if "Your browser hit a snag and we need to make sure you're not a bot." in driver.page_source:
            logger.info("Page Blocked")
            block_page = True
            return c_avg(low_dict), c_avg(low_dict_resl), c_avg(high_dict), c_avg(
                high_dict_resl), error, no_best, block_page
    except Exception as e:
        error = False
    
    try:
        ticket_items = wait.until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, 'li[data-bdd^="quick-picks-list-item-"]')))

        
        # Loop through each ticket item and extract both the Ticket Type text and Price
        for item in ticket_items:
            try:
                try:
                    # Locate the Ticket Type span within the current ticket item
                    ticket_type_element = item.find_element(By.CSS_SELECTOR, 'div[data-bdd="branding-ticket-text"]')
                    ticket_type_text = ticket_type_element.text
                except:
                    ticket_type_text = item.find_element(By.CSS_SELECTOR, 'span[data-bdd="quick-picks-resale-branding"]').text
                
                # Locate the Price button within the current ticket item
                price_element = item.find_element(By.CSS_SELECTOR, 'button[data-bdd="quick-pick-price-button"]')
                price_text = price_element.text
                if "Resale" in ticket_type_text:
                    low_dict_resl.append(float(strip_non_digit_decimal(price_text)))
                else:
                    low_dict.append(float(strip_non_digit_decimal(price_text)))
                
            except Exception as e:
                dummy = 0
                # logger.info("Could not locate Ticket Type or Price for an item:", e)
                
    except Exception as e:
        error = True
        # logger.debug(f"Error: low price tickets not available.{e}")
        return c_avg(low_dict), c_avg(low_dict_resl), c_avg(high_dict), c_avg(high_dict_resl), error, no_best, block_page


    try:
        # Wait until the "BEST SEATS" button is clickable
        best_seats_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'span[data-bdd="quick-picks-sort-button-best"]'))
        )
        # Click the "BEST SEATS" button
        best_seats_button.click()
        
        # logger.info("Clicked the BEST SEATS button.")
    
    except Exception as e:
        no_best = True
        return c_avg(low_dict), c_avg(low_dict_resl), c_avg(high_dict), c_avg(high_dict_resl), error, no_best, block_page
        # logger.info("Could not click the BEST SEATS button:", e)

    
    try:
        ticket_items = wait.until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, 'li[data-bdd^="quick-picks-list-item-"]')))
        
        for item in ticket_items:
            try:
                try:
                    # Locate the Ticket Type span within the current ticket item
                    ticket_type_text = item.find_element(By.CSS_SELECTOR, 'span[data-bdd="quick-picks-resale-branding"]').text
                except:
                    ticket_type_element = item.find_element(By.CSS_SELECTOR, 'div[data-bdd="branding-ticket-text"]')
                    ticket_type_text = ticket_type_element.text
                # Locate the Price button within the current ticket item
                price_text = item.find_element(By.CSS_SELECTOR, 'button[data-bdd="quick-pick-price-button"]').text
                if "Resale" in ticket_type_text:
                    high_dict_resl.append(float(strip_non_digit_decimal(price_text)))
                else:
                    high_dict.append(float(strip_non_digit_decimal(price_text)))
                
            except Exception as e:
                dummy = 0
                # logger.info("Could not locate Ticket Type or Price for an item:", e)

    except Exception as e:
        no_best = True
        # logger.info("Error: high price tickets not available.", e)

    return c_avg(low_dict), c_avg(low_dict_resl), c_avg(high_dict), c_avg(high_dict_resl), error, no_best, block_page



# In[4]:
def run_selenium():
    driver_list = []
    driver, driver_dict = initialize_driver()
    driver_list.append(driver_dict)
    df = get_price_df()
    min_wait = 2
    max_wait = 15
    df['Stan_Min'] = None
    df['Stan_Max'] = None
    df['Resl_Min'] = None
    df['Resl_Max'] = None
    count = 0
    epoch_count = 0
    error = 0
    epoch_error = 0
    only_low = 0
    epoch_only_low = 0
    blocked_sequence = 0
    error_sequence = 0
    driver_count = 0
    start_time = time.time()
    for index, row in df.iterrows():
        # logger.debug(f"Instance: {index}")
        url = row.get('url', None)
        if url and 'ticketmaster' in url:
            stan_min, resl_min, stan_max, resl_max, error_flag, no_best, block_page = process(url, driver)
            df.at[index, 'Stan_Min'] = stan_min
            df.at[index, 'Stan_Max'] = stan_max
            df.at[index, 'Resl_Min'] = resl_min
            df.at[index, 'Resl_Max'] = resl_max
        else:
            block_page = False
            error_flag = True
            no_best = False
            df.at[index, 'Stan_Min'] = 0
            df.at[index, 'Stan_Max'] = 0
            df.at[index, 'Resl_Min'] = 0
            df.at[index, 'Resl_Max'] = 0

        # Logic to reset the chromedriver instance if the page has been blocked more than once
        if block_page:
            blocked_sequence += 1
            if blocked_sequence > 1:
                driver, driver_list = reset_driver(driver, driver_list, driver_count, min_wait, max_wait)
                blocked_sequence = 0
                driver_count = 0
        else:
            blocked_sequence = 0

        driver_count += 1
        count += 1
        epoch_count += 1
        # Logic to count errors
        if error_flag:
            error_sequence += 1
            # Logic to reset the driver if there are consecutive errors on more than 10 pages
            if error_sequence > 10:
                driver, driver_list = reset_driver(driver, driver_list, driver_count, min_wait, max_wait)
                error_sequence = 0
                driver_count = 0
            error += 1
            epoch_error += 1
        else:
            error_sequence = 0
        # Logic to count certain page conditions
        if no_best:
            only_low += 1
            epoch_only_low += 1
        if count % 100 == 0:
            logger.info(f"100 complete, error rate: {error / count}, only lowest rate: {only_low / count}  this batches error: {epoch_error / epoch_count}, this batches only_low: {epoch_only_low / epoch_count}")
            epoch_count = 0
            epoch_error = 0
            epoch_only_low = 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(df.head())

    if count > 0:
        logger.info(f"Error %: {error / count}")
        logger.info(f"No_High %: {only_low / count}")
    logger.info(f"Elapsed Minutes: {elapsed_time/60}")
    prev_driver = driver_list.pop()
    prev_driver['driver_count'] = driver_count
    driver_list.append(prev_driver)
    driver.quit()
    df_drivers = pd.DataFrame(driver_list)
    df_drivers.to_csv(r"csvs\driver_list.csv", index = False)
    db = DB_API()
    db.update_prices(df)
    db.insert_records()
    df.to_csv(r"csvs\price_update.csv", index = False)
    db.close()







