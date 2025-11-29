import requests
from bs4 import BeautifulSoup
# import boto3
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger


class DailyScraper:
    def __init__(self, url = None):
        # Define URLs and output file paths
        self.url = url
        self.urls = {
            "NFL": "https://www.espn.com/nfl/fpi",
            "NBA": "https://www.espn.com/nba/bpi",
            "NHL": "https://www.espn.com/nhl/standings/_/group/league",
            "Artist100": "https://www.billboard.com/charts/artist-100/"
        }
        self.output_files = {
            "NFL": r"csvs/nfl_power_rankings.csv",
            "NBA": r"csvs/nba_power_rankings.csv",
            "NHL": r"csvs/nhl_power_rankings.csv",
            "Artist100": r"csvs/artist_100_rankings.csv"
        }
        # self.s3_client = boto3.client('s3')
        # self.bucket_name = 'ticket-generator-storage'
    
    def fetch_page(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }    # Can be adjusted to the referring page
        

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.content
    
    def parse_and_save(self, html, category):
        # Parse the HTML and extract rankings based on the category
        soup = BeautifulSoup(html, "html.parser")
        data = []
        
        # Parsing logic based on the category
        if category == "NFL" or category == "NBA":
            # ESPN structure
            table = soup.find('table', class_='Table Table--align-right Table--fixed Table--fixed-left')
            for ind, row in enumerate(table.find_all('tr', class_='Table__TR Table__TR--sm Table__even')):
                img_tag = row.find('img')
                alt_text = img_tag['alt'] if img_tag and img_tag.has_attr('alt') else None
                rank = ind + 1
                data.append([alt_text, rank])

        elif category == 'NHL':
            # ESPN structure
            table = soup.find('table', class_='Table Table--align-right Table--fixed Table--fixed-left')
            for ind, row in enumerate(table.find('tbody').find_all('tr')):
                span = row.find('span', class_='hide-mobile')
                text = span.find('a').text
                rank = ind + 1
                data.append([text, rank])
        
        elif category == "Artist100":
            # Billboard structure
             # Modify based on actual HTML structure
            rows = soup.find_all('div', class_='o-chart-results-list-row-container')
            rank = 1
            for row in rows:
                name = row.find('h3')
                text = name.text.strip()
                if text != "Imlogger.info/Promotion Label:" and text != "Gains in Weekly Performance" and text != "Additional Awards":
                    data.append([text, rank])
                    rank += 1
        # Save data to CSV
        self.save_to_path(data, self.output_files[category])

    def save_to_path(self, data, file_path):
        # Overwrite daily
        df = pd.DataFrame(data, columns=["Name", "Rank"])
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")

    def run_daily_scrape(self):
        # Run the scraping process for each URL
        for category, url in self.urls.items():
            try:
                logger.info(f"Scraping {category} data...")
                html = self.fetch_page(url)
                self.parse_and_save(html, category)
            except Exception as e:
                logger.error(f"Failed to scrape {category}: {e}")
    
    # def check_table(self, table):
    #     path = self.output_files[table]
    #     response = self.s3_client.get_object(Bucket=self.bucket_name, Key=path)
    #     csv_content = response['Body'].read().decode('utf-8')
    #     df = pd.read_csv(io.StringIO(csv_content))
    #     logger.info(df.head())
    
    def get_table(self, table):
        path = self.output_files[table]
        df = pd.read_csv(path)
        return df

    def scrape_url(self):
        html = self.fetch_page(self.url)
        soup = BeautifulSoup(html, "html.parser")
        return soup