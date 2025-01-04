#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db_api import DB_API
from logger import logger
import tempfile
import shutil

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import re
import time
from multiprocessing import Pool


# In[2]:
def get_price_df():
    db = DB_API()
    df = db.get_prices()
    logger.info(df.head())
    logger.info(len(df))
    db.close()
    return df


# In[3]:
def initialize_driver():
    temp_dir = tempfile.mkdtemp()
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument(f"--user-data-dir={temp_dir}")
    # chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36")
    # chrome_options.add_argument("--accept-language=en-US,en;q=0.9")
    # chrome_options.add_argument("referer=https://www.google.com/")
    # chrome_options.add_argument("--disable-dev-shm-usage")
    # chrome_options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver, temp_dir


def strip_non_digit_decimal(input_string):
    return re.sub(r'[^0-9.]+', '', input_string)

def c_avg(numbers):
    return sum(numbers) / len(numbers) if numbers else 0  # Avoid division by zero if list is empty

def process(row):
    driver, temp_dir = initialize_driver()
    url = row['url']
    error = False
    wait = WebDriverWait(driver, 4)
    low_dict = []
    low_dict_resl = []
    high_dict = []
    high_dict_resl = []
    try:
        driver.get(str(url))
    except:
        error = True
        driver.quit()
        shutil.rmtree(temp_dir)
        return c_avg(low_dict), c_avg(low_dict_resl), c_avg(high_dict), c_avg(high_dict_resl), error
    try:
        accept_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-bdd="accept-modal-accept-button"]'))
        )
        accept_button.click()
    except Exception as e:
        # logger.info("Error: Button not found or could not be clicked.", e)
        error = False
        # Find all the ticket items in the quick-picks list
    
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
        driver.quit()
        shutil.rmtree(temp_dir)
        return c_avg(low_dict), c_avg(low_dict_resl), c_avg(high_dict), c_avg(high_dict_resl), error

        # logger.info("Error: low price tickets not available.", e)

    try:
        # Wait until the "BEST SEATS" button is clickable
        best_seats_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'span[data-bdd="quick-picks-sort-button-best"]'))
        )
        # Click the "BEST SEATS" button
        best_seats_button.click()
        
        # logger.info("Clicked the BEST SEATS button.")
    
    except Exception as e:
        error = True
        driver.quit()
        shutil.rmtree(temp_dir)
        return c_avg(low_dict), c_avg(low_dict_resl), c_avg(high_dict), c_avg(high_dict_resl), error

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
        error = True
        # logger.info("Error: high price tickets not available.", e)
    driver.quit()
    shutil.rmtree(temp_dir)
    return c_avg(low_dict), c_avg(low_dict_resl), c_avg(high_dict), c_avg(high_dict_resl), error



# In[4]:
def run_selenium(df):

    df['Stan_Min'] = None
    df['Stan_Max'] = None
    df['Resl_Min'] = None
    df['Resl_Max'] = None
    count = 0
    epoch_count = 0
    error = 0
    epoch_error = 0
    start_time = time.time()
    rows = df.to_dict(orient='records')
    with Pool(processes=6) as pool:
        for index, (stan_min, resl_min, stan_max, resl_max, error_flag) in enumerate(pool.imap(process, rows)):
            df.at[index, 'Stan_Min'] = stan_min
            df.at[index, 'Stan_Max'] = stan_max
            df.at[index, 'Resl_Min'] = resl_min
            df.at[index, 'Resl_Max'] = resl_max
            count += 1
            epoch_count += 1
            if error_flag:
                error += 1
                epoch_error += 1

            if count % 100 == 0:
                logger.info(
                    f"100 complete, error rate: {error / count}, this batches error: {epoch_error / epoch_count}")
                epoch_count = 0
                epoch_error = 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(df.head())
    if count > 0:
        logger.info(f"Error %: {error / count}")
    logger.info(f"Elapsed Minutes: {elapsed_time/60}")

    try:
        db = DB_API()
        db.update_prices(df)
        db.insert_records()
        db.close()
    except Exception as e:
        logger.error(f"Error updating db: {e}, saving to csv")
        df.to_csv(r"csvs\price_update.csv", index=False)








