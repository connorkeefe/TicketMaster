### About this project

This project is designed to be run daily. The goal of the project in its current state was to accumulate live price data of different 
events on ticketmaster in Canada and the US over time along with other information on the event partcipants be it sports 
teams or artists. This project leverages ticketmasters discovery API to retrieve most of the information
on the different events, for the live price data two different strategies are laid out, one is brute force selenium webscraping
the other is reverse engineering ticketmasters inventory API.

The end goal of this project was to gather enough live price data (at least 6 months worth) and train a temporal fusion transformer model
on the data in order to predict future price movements of tickets based on all available data.

**Disclaimer:** I don't condone going against ticketmaster terms and conditions specifically with regard to their data privacy, this project is
purely for educational purposes.

### Environment setup

#### Python requirements
Navigate to the project folder:

Python 3.11.11

```pip install requirements.txt```

#### TicketMaster API Key
1. Go to the discovery API page:
https://developer.ticketmaster.com/products-and-docs/apis/discovery-api/v2/
2. Create an account and then click **Add A New App** on your profile
3. Under this app find the **Consumer Key**
4. Set this as an environment variable 
```export TM_API_KEY="your-api-key-here"```

** Important to note: ticketmaster only allows 5000 daily calls on the discovery API daily so this script is designed to make around 4000 calls per day

#### AWS RDS DB Setup
** Feel free to reconfigure the code to write to a local db file, 
should be simple change to init() in ```db/db_api.py```

1. Create an RDS db on AWS, choose from free tier -> Engine: PostgreSQL, Size: db.t4g.micro
2. Create a password for this db, remember this
3. Set the Password and Endpoint (in Connectivity & Security tab) of the db as environment variables: 

```export DB_HOST="your-endpoint-here"```
```export DB_PWD="your-password-here"```

The db username and db name should remain unchanged if a postgres db is used in the free tier but these values can be modified in ```db/db_api.py```

#### Selenium Setup
1. in ``DailyPriceScraper/selenium_runner.py`` update the DRIVER_PATH to the absolute path of the chrome driver in your project folder
2. For Mac there is no extension on chromedriver, for Windows you must add the .exe extension
3. You may also have to download a different chromedriver binary depending on what version of chrome your computer is running, can check this at: chrome://settings/help
4. Can download a chrome driver matching your chrome version and computer architecture: https://developer.chrome.com/docs/chromedriver/downloads

** in ```DailyPriceScraper/selenium_runner.py``` you can also play around with the driver settings, if you want to see the driver when it runs then comment out the 'headless' option in the initialize driver function

** in ```DailyPriceScraper/selenium_runner.py``` there is also ```USERNAME, PASSWORD, and ENDPOINT``` constants that can be set if you wish to use a proxy, you would also have to remove the comment from line 125 and add proxies as a parameter in line 126  

### Usage

With the environment setup this project is run with: ```python runner.py```

On first run you must set the ```CREATE``` constant in ```DailyAPIget/lambda_function.py``` to ```True```
This will create the tables in the database. There are more toggles in this file like ```DELETE``` which will delete all tables.

Logs can be found in ```app.log```

```scrape_inventory.run_inventory_api()``` is commented out but is another more efficient way to scrape the ticket price data for each event

```pull_training_data.pull_train_data()``` is commented out but this was the start of my script that would be used to pull the data for model training purposes

