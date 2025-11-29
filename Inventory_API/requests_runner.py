import json
from curl_cffi import requests
from run_flags import SCRAPE_TICKETS
from Inventory_API.parse_utils import EVENTID, URL
from Inventory_API.camoufox_runner import StealthWeb
from logger import logger

# PROXY = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text"
PROXY = None

def get_response(event):
    id = event.get(EVENTID, None)
    logger.info(id)
    page = event.get(URL, None)
    logger.info(page)
    tickets_resp, tickets_body = None, None
    if page:
        if "www.ticketmaster.com" not in page:
            logger.info(f"No Ticketmaster Link: {id}")
            return False, None, None, None, False, None, None
        stealth_web = StealthWeb(proxy=PROXY)
        if SCRAPE_TICKETS:
            (response, body), (tickets_resp, tickets_body) = stealth_web.get_response_both(page,
                                                                                           primary_substr='inventorytypes%20offertypes%',
                                                                                           secondary_substr='inventorytypes%20offer&q=available&show=listpricerange',
                                                                                           wait_secondary=True)
        else:
            logger.info(f"Running get_response {id}")
            response, body = stealth_web.get_response(page)

        # logger.info(body)
        if response and len(body) > 1:
            if tickets_resp and len(tickets_body) > 1:
                return True, response, body, stealth_web.headers, True, tickets_resp, tickets_body
            else:
                return True, response, body, stealth_web.headers, False, tickets_resp, tickets_body
        elif tickets_resp and len(tickets_body) > 1:
            return False, None, None, stealth_web.headers, True, tickets_resp, tickets_body
    return False, None, None, None, False, None, None


def get_response_requests(event, headers):
    # time.sleep(random.choice([1,2]))
    id = event.get(EVENTID, None)
    logger.info(id)
    page = event.get(URL, None)
    logger.info(page)
    try:
        if id:
            url = f'https://offeradapter.ticketmaster.com/api/ismds/event/{id}/facets?q=available&by=inventorytypes%20offertypes%20area%20tickettype%20priceLevels&show=listpricerange&resaleChannelId=internal.ecommerce.consumer.desktop.web.browser.ticketmaster.us&apikey=b462oi7fic6pehcdkzony5bxhe&apisecret=pquzpfrfz7zd2ylvtz3w5dtyse&embed=area&embed=tickettype'
            proxies = {"http": PROXY, "https": PROXY} if PROXY else None
            response = requests.get(url, headers=headers, proxies=proxies)
            data = json.loads(response.content.decode('utf-8'))
            if response.status_code == 404:
                page = event.get(URL, None)
                if page:
                    id = page.split("/")[-1]
                    url = f'https://offeradapter.ticketmaster.com/api/ismds/event/{id}/facets?q=available&by=inventorytypes%20offertypes%20area%20tickettype%20priceLevels&show=listpricerange&resaleChannelId=internal.ecommerce.consumer.desktop.web.browser.ticketmaster.us&apikey=b462oi7fic6pehcdkzony5bxhe&apisecret=pquzpfrfz7zd2ylvtz3w5dtyse&embed=area&embed=tickettype'
                    response = requests.get(url, headers=headers, proxies=proxies)
                    data = json.loads(response.content.decode('utf-8'))
            return True, response, data
        return False, None, None
    except Exception as e:
        logger.error(f"Error processing request for id: {id}, {e}")
        return False, None, None



def get_ticket_response_requests(event, headers):
    # time.sleep(random.choice([1,2]))
    id = event.get(EVENTID, None)
    logger.info(id)
    page = event.get(URL, None)
    logger.info(page)
    try:
        if id:
            url = f'https://offeradapter.ticketmaster.com/api/ismds/event/{id}/facets?apikey=b462oi7fic6pehcdkzony5bxhe&apisecret=pquzpfrfz7zd2ylvtz3w5dtyse&by=inventorytypes%20offer&q=available&show=listpricerange&embed=offer&resaleChannelId=internal.ecommerce.consumer.desktop.web.browser.ticketmaster.us'
            proxies = {"http": PROXY, "https": PROXY} if PROXY else None
            response = requests.get(url, headers=headers, proxies=proxies)
            data = json.loads(response.content.decode('utf-8'))
            if response.status_code == 404:
                page = event.get(URL, None)
                if page:
                    id = page.split("/")[-1]
                    url = f'https://offeradapter.ticketmaster.com/api/ismds/event/{id}/facets?apikey=b462oi7fic6pehcdkzony5bxhe&apisecret=pquzpfrfz7zd2ylvtz3w5dtyse&by=inventorytypes%20offer&q=available&show=listpricerange&embed=offer&resaleChannelId=internal.ecommerce.consumer.desktop.web.browser.ticketmaster.us'
                    response = requests.get(url, headers=headers, proxies=proxies)
                    data = json.loads(response.content.decode('utf-8'))
            return True, response, data
        return False, None, None
    except Exception as e:
        logger.error(f"Error processing get_ticket request for id: {id}, {e}")
        return False, None, None