import requests
import uuid
import json
import random
from datetime import datetime, timedelta
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db_api import DB_API
from DailyPriceScraper.selenium_runner import get_cookies, USER_AGENTS
from logger import logger
import time

def get_price_id_df():
    db = DB_API()
    df = db.get_event_ids_prices()
    logger.info(df.head())
    logger.info(len(df))
    db.close()
    return df

def average_of_list(data):
    # Filter out None and convert valid float strings to floats
    valid_numbers = [float(x) for x in data if x is not None and str(x).replace('.', '', 1).isdigit()]

    if not valid_numbers:  # Check if the list of valid numbers is empty
        return None  # Return None if there are no valid numbers

    # Calculate the average
    return sum(valid_numbers) / len(valid_numbers)


def parse_facet(lst):
    resale_min = []
    resale_max = []
    standard_min = []
    standard_max = []
    count = 0
    for listing in lst:
        price_dict = listing.get('listPriceRange', [''])[0]
        min_price = price_dict.get('min', None)
        max_price = price_dict.get('max', None)
        count += listing.get('count', 0)
        if 'resale' in listing.get('inventoryTypes', [''])[0]:
            resale_min.append(min_price)
            resale_max.append(max_price)
        else:
            standard_min.append(min_price)
            standard_max.append(max_price)

    return average_of_list(standard_min), average_of_list(standard_max), average_of_list(resale_min), average_of_list(
        resale_max), count


def run_inventory_api():

    df = get_price_id_df()
    df['Stan_Min'] = None
    df['Stan_Max'] = None
    df['Resl_Min'] = None
    df['Resl_Max'] = None
    count = 0
    error_count = 0
    start_time = time.time()
    # Swap out when necessary
    url = 'https://www.ticketmaster.com/event/Z7r9jZ1A78U7V'
    # url = 'https://www.google.com/'
    cookie_header = get_cookies(url)
    for index, event in df.iterrows():
        time.sleep(random.choice([1,2]))
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "if-modified-since": "Mon, 9 Dec 2024 00:54:46 GMT",
            "if-none-match": "W/\"02cc090781d33fbdc9857d426836d5299\"",
            "priority": "u=1, i",
            "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "tmps-correlation-id": str(uuid.uuid4()),
            # "cookie": "LANGUAGE=en-us; reese84=3:zbCXxFZPWOmEfhJuERJiQg==:cOaXpIMLFTXf4v2wUOmXKbBJ32bYKG3CsVV1beiPqlmrBDzEiIZ+PlqNubxtZASmvuvDwhvXJHKp+rQLfx3owB9qmYYCN0xehQ++Zb3G2e8JmAa6ZQPQT8+5do4RgAaRRo1w1JQi7VeN60NvtcMfvE/KIBYF0O+x9mPvnGWRt/S14B/U3UONbx4UZXf08g2AG7nsUXORinE/RZrBNW1I8hRNDMNpozLAln43v1Av0tLHVwhKPefP3ctGnKQUoPjFmZZsWFHIkj+gaaHPpGBX9iSrp9DpVAnKpXYNJW4V+vgtGKZok3TOkg7/bGYQcTnkm0ZuChh4fMxRoQSLx/S4E11f26XC25oOEuh+KckKIxie790NCx2Xxy1FTigH/BNQTOE2NYHzWrjtfGXQD2j1UY5XTzTWC4+6U9bwCyJtyvypev5BbsQEoEHnTqTdtACyCgSlVlG9J7tGuP/8g9S3fQ==:VDTCFfnmg+bVt5+avbWhDqd1m2hXz2fJyU+ap8Hv6No=; mt.v=2.8639196.1731554697371; _gcl_au=1.1.1285400550.1731623084; _gac_UA-60025178-1=1.1731623086.CjwKCAiA3Na5BhAZEiwAzrfagIWb_Bj1bBdXDahYNdtHwpqpN-15v4FUwFQ4msqs7CFyGlwsbZ0nCRoCXuEQAvD_BwE; _au_1d=AU1D02000017316231245XF2ELUQUEPD; _gac_UA-60025178-2=1.1731623086.CjwKCAiA3Na5BhAZEiwAzrfagIWb_Bj1bBdXDahYNdtHwpqpN-15v4FUwFQ4msqs7CFyGlwsbZ0nCRoCXuEQAvD_BwE; _scid=4ebECHa9aOL8_5wxJnsR0gb7gIOvs7GY; _gac_UA-204680695-1=1.1731623086.CjwKCAiA3Na5BhAZEiwAzrfagIWb_Bj1bBdXDahYNdtHwpqpN-15v4FUwFQ4msqs7CFyGlwsbZ0nCRoCXuEQAvD_BwE; _cs_c=0; ken_gclid=CjwKCAiA3Na5BhAZEiwAzrfagIWb_Bj1bBdXDahYNdtHwpqpN-15v4FUwFQ4msqs7CFyGlwsbZ0nCRoCXuEQAvD_BwE; _tt_enable_cookie=1; _ttp=n6aalZLAJJD-glw_sl2QNBzkTa4.tt.1; OptanonGroups=,C0001,C0003,C0002,C0004,; __qca=P0-286094125-1731623134478; AMCV_F75C3025512D2C1D0A490D44%40AdobeOrg=179643557%7CMCIDTS%7C20042%7CMCMID%7C85090971473369302720654915176262655627%7CMCAAMLH-1732227936%7C7%7CMCAAMB-1732227936%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1731630341s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C5.5.0; _ga_61CBJ21K20=GS1.1.1731623137.1.1.1731623142.0.0.0; kndctr_F75C3025512D2C1D0A490D44_AdobeOrg_identity=CiY4NTA5MDk3MTQ3MzM2OTMwMjcyMDY1NDkxNTE3NjI2MjY1NTYyN1IQCP_doOayMhgBKgNWQTYwA_AB_92g5rIy; _ga_FVWZ0RM4DH=GS1.1.1731623151.1.0.1731623151.60.0.0; NDMA=200; BID=lQwUMK12USqGIUEaXJh1Xk9SuvZIsGx0L2qEZa05eTD98NLawNRjtZC82_PauBV5noWuqq0mDpJ615Wz; SID=SkC6CFjJrJ6rAIpsKQQhW_gKOhCGj5TpRJx-2gWOH4Hgyc2rCntHMLOhjKSekON8udTJey2DggwJ3Y2B; eps_sid=c1d74c570c3241598f040e27f64eb0bb7d735486; TMUO=east_xfO55xblOX2CmZEeyCJUHVME8j9qqYagCDWmKVfGxpU=; _gcl_aw=GCL.1733869092.CjwKCAiA6t-6BhA3EiwAltRFGHO-yONKdCzO83r-GFOaW-bRgqXJhwThFJzxM8Q8V9pjoa8V3A7h9RoCMAcQAvD_BwE; _gcl_dc=GCL.1733869092.CjwKCAiA6t-6BhA3EiwAltRFGHO-yONKdCzO83r-GFOaW-bRgqXJhwThFJzxM8Q8V9pjoa8V3A7h9RoCMAcQAvD_BwE; _gid=GA1.2.2072632825.1733869827; TM_PIXEL={\"_dvs\":\"0:m4j1bt8c:BXDwf21hjCvvNVZ9p0F~zfAe9wCbsOjD\",\"_dvp\":\"0:m3hvp6th:WKr9IlRfeyS08GQbLGu5IW37scjUY7f9\"}; mt.pc=2.1; mt.g.2f013145=2.8639196.1731554697371; _ScCbts=%5B%221%3Bchrome.2%3A2%3A5%22%5D; _sctr=1%7C1733806800000; s_fid=6DFE90D7F1C50F0F-38581DD19A4FD333; s_cc=true; nbatag_main_v_id=0193b2b62d950014d4e3f0f4779205075004806d00bd0; AAMC_nba_0=REGION%7C7; aam_uuid=85291444949868419510674766562913165876; AMCVS_248F210755B762187F000101%40AdobeOrg=1; AMCV_248F210755B762187F000101%40AdobeOrg=179643557%7CMCIDTS%7C20068%7CMCMID%7C85260224653539864000671704185435577355%7CMCAAMLH-1734474904%7C7%7CMCAAMB-1734474904%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1733877307s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C5.5.0; s_ppv=%5B%5BB%5D%5D; s_ips=0; s_tp=0; _ga_RY9H8ZP8MH=GS1.1.1733870106.1.0.1733870109.57.0.0; AMCVS_C1A25E0B5B89E6F90A495CEA%40AdobeOrg=1; AMCV_C1A25E0B5B89E6F90A495CEA%40AdobeOrg=1585540135%7CMCMID%7C85493014186794315430695211854905613340%7CMCAAMLH-1734474910%7C7%7CMCAAMB-1734474910%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1733877310s%7CNONE%7CvVersion%7C4.4.0; s_plt=0.08; s_pltp=Ticketmaster%3A%20Onsale%3A%20; amp_297f60=B1vwU6yW0fHwZiMugr8zLg...1iepbcj23.1iepbcj2q.1.1.2; _pn=eyJzdWIiOnsidWRyIjowLCJpZCI6ImNnblZZbU9Zb3NOTG1vaVZUOWc4eDBkUUVaS294UTNmIiwic3MiOi0xfSwibHVhIjoxNzMzODcwMTMzOTUyfQ; _ga_DGMCRNLESF=GS1.1.1733870179.1.1.1733870873.59.0.0; _pin_unauth=dWlkPU1qQXpZVGt5TlRNdFkyUXlNUzAwTldSbUxUZ3hZV1F0WWpjMk1XUmtNbUZrWkRJMg; _ga_TEX9PJVHCM=GS1.1.1733870983.1.0.1733870983.0.0.0; _ga_JV3ETVD344=GS1.1.1733870983.1.0.1733870983.0.0.0; _ga_78QWFWEZB7=GS1.1.1733870983.1.0.1733870983.0.0.0; _ga_CG2TMSX6FC=GS1.1.1733870982.1.0.1733870988.0.0.0; _ga_176J45V845=GS1.1.1733870982.1.0.1733870988.0.0.0; _ga_H69P9DGLEF=GS1.1.1733870984.1.1.1733870990.0.0.0; _ga_CCZ5558LN7=GS1.1.1733871161.1.1.1733871813.0.0.0; _ga_XZHCNEQSGP=GS1.1.1733871161.1.1.1733871813.0.0.0; _ga_SDMPWV97S3=GS1.1.1733871162.1.1.1733871814.0.0.0; tmpt=0:5a2972d556000000:1733875530:34a72655:b0c954b3a761af5649f2569797f1dec3:f1f28bd82e1e87c65be570840bbb042b5ac3d71d6a02838980f1b267cb02cd8c; _dc_gtm_UA-60025178-1=1; _scid_r=92bECHa9aOL8_5wxJnsR0gb7gIOvs7GY4WMuHg; _rdt_uuid=1733869866836.d344ac9b-400a-486f-a94a-c011927ecf6e; _dc_gtm_UA-60025178-2=1; _gat_tracker28719257=1; nbatag_main__sn=2; nbatag_main__se=1%3Bexp-session; nbatag_main__ss=1%3Bexp-session; nbatag_main__st=1733877355473%3Bexp-session; nbatag_main_ses_id=1733875555473%3Bexp-session; nbatag_main__pn=1%3Bexp-session; _ga=GA1.1.1537119387.1731623085; _cs_cvars=%7B%221%22%3A%5B%22Page%20Name%22%2C%22TM_US%3A%20CCP%20EDP%3A%20RS%3A%20Onsale%22%5D%2C%222%22%3A%5B%22Page%20Type%22%2C%22CCP%20EDP%3A%20Onsale%22%5D%2C%223%22%3A%5B%22Modules%20Available%22%2C%22EDP_RseCalVfs%22%5D%2C%224%22%3A%5B%22Platform%22%2C%22ccp-edp%22%5D%2C%225%22%3A%5B%22Login%20Status%22%2C%22Not%20Logged%20In%22%5D%2C%226%22%3A%5B%22Major%20Category%22%2C%22Sports%22%5D%2C%227%22%3A%5B%22Minor%20Category%22%2C%22Basketball%22%5D%2C%228%22%3A%5B%22Artist%20ID%22%2C%22806042%22%5D%2C%229%22%3A%5B%22Artist%20Name%22%2C%22Washington%20Wizards%22%5D%2C%2210%22%3A%5B%22Venue%20ID%22%2C%22172453%22%5D%2C%2211%22%3A%5B%22Event%20ID%22%2C%221500610EEB594E74%22%5D%2C%2212%22%3A%5B%22Event%20Date%22%2C%224%2F3%2F2025%22%5D%2C%2213%22%3A%5B%22EDP%20Page%20Type%22%2C%22CCP%20EDP%3A%20SIM%22%5D%2C%2214%22%3A%5B%22Event%20Type%22%2C%22STANDARD%22%5D%7D; _cs_id=56ada184-2e7f-a4ba-de91-604f449afd28.1731623132.3.1733875558.1733875339.1723134244.1765787132221.1; _ga_MCB7B7H198=GS1.1.1733875556.2.0.1733875560.0.0.0; _ga_J43DPCSGK1=GS1.1.1733875556.1.0.1733875560.0.0.0; _ga_37ZH6WRP1C=GS1.1.1733875557.2.0.1733875560.0.0.0; _ga_YE3V7130LW=GS1.2.1733875561.2.0.1733875561.0.0.0; _uetsid=75c141d0b74611ef8bf0d7d046387f79|17nfob3|2|frm|0|1805; _cs_s=2.5.1.1733877362268; OptanonConsent=isGpcEnabled=0&datestamp=Tue+Dec+10+2024+19%3A06%3A02+GMT-0500+(Eastern+Standard+Time)&version=202408.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=bac43b98-58e5-41e5-9906-df131a9ea975&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0003%3A1%2CC0002%3A1%2CC0004%3A1&AwaitingReconsent=false; _uetvid=5a156f80a2d711ef964839658fa60f51|dxqpa2|1733875563779|1|1|bat.bing.com/p/insights/c/f; amp_ff147e=dAJR4QMDc7zPrCsNDS1_iN...1iepgivhi.1iepgivi4.1.1.2; _ga_C1T806G4DF=GS1.1.1733875433.3.1.1733875572.24.0.0; _ga_H1KKSGW33X=GS1.1.1733875536.3.0.1733875572.24.0.0",
            "cookie": cookie_header,
            "Referer": "https://www.ticketmaster.ca/",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "User-Agent": random.choice(USER_AGENTS)
        }
        id = event.get('eventid', None)
        if id:
            url = f'https://offeradapter.ticketmaster.com/api/ismds/event/{id}/facets?q=available&by=inventorytypes%20offertypes%20area%20tickettype%20priceLevels&show=listpricerange&resaleChannelId=internal.ecommerce.consumer.desktop.web.browser.ticketmaster.us&apikey=b462oi7fic6pehcdkzony5bxhe&apisecret=pquzpfrfz7zd2ylvtz3w5dtyse&embed=area&embed=tickettype'
            response = requests.get(url, headers=headers)
            data = json.loads(response.content.decode('utf-8'))
            if response.status_code == 404:
                page = event.get('url', None)
                if page:
                    id = page.split("/")[-1]
                    url = f'https://offeradapter.ticketmaster.com/api/ismds/event/{id}/facets?q=available&by=inventorytypes%20offertypes%20area%20tickettype%20priceLevels&show=listpricerange&resaleChannelId=internal.ecommerce.consumer.desktop.web.browser.ticketmaster.us&apikey=b462oi7fic6pehcdkzony5bxhe&apisecret=pquzpfrfz7zd2ylvtz3w5dtyse&embed=area&embed=tickettype'
                    response = requests.get(url, headers=headers)
                    data = json.loads(response.content.decode('utf-8'))
            if response.status_code == 200:
                logger.info(response)
                facets = data.get("facets", [])
                logger.info(facets)
                if len(facets) > 0:
                    new_url = event.get('url', None)
                    if new_url:
                        url = new_url
                    stan_min, stan_max, resl_min, resl_max, count = parse_facet(facets)
                    logger.info(f"Results: {stan_min}, {stan_max}, {resl_min}, {resl_max}, {count}")
                    df.at[index, 'Stan_Min'] = stan_min
                    df.at[index, 'Stan_Max'] = stan_max
                    df.at[index, 'Resl_Min'] = resl_min
                    df.at[index, 'Resl_Max'] = resl_max
                else:
                    df['Stan_Min'] = None
                    df['Stan_Max'] = None
                    df['Resl_Min'] = None
                    df['Resl_Max'] = None

                logger.info(event.get('url', None))
            elif response.status_code == 403:
                logger.info(response)
                logger.info(response.content)
                logger.info(event.get('url', None))
                cookie_header = get_cookies(url)
                error_count += 1
                df['Stan_Min'] = None
                df['Stan_Max'] = None
                df['Resl_Min'] = None
                df['Resl_Max'] = None
            else:
                logger.info(response)
                logger.info(response.content)
                logger.info(event.get('url', None))
                error_count += 1
                df['Stan_Min'] = None
                df['Stan_Max'] = None
                df['Resl_Min'] = None
                df['Resl_Max'] = None
            count += 1
            if count % 10 == 0:
                logger.info(
                    f"10 complete, error rate: {error_count / count}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(df.head())
    if count > 0:
        logger.info(f"Error %: {error_count / count}")
    logger.info(f"Elapsed Minutes: {elapsed_time / 60}")
    db = DB_API()
    db.update_prices(df)
    db.insert_records()
    df.to_csv(r"csvs\price_update.csv", index=False)
    db.close()


