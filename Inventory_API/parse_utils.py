import json
import ast
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any
from DailyAPIget.api_handler import generate_timestamped_uuid
from logger import logger
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



URL = 'URL'
EVENTID = 'EventID'

### Event Prices Parsing ####

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
    resale_count = 0
    standard_count = 0
    count = 0
    for listing in lst:
        price_list = listing.get('listPriceRange', [])
        price_dict = price_list[0] if price_list else {}
        min_price = price_dict.get('min', None)
        max_price = price_dict.get('max', None)
        count = listing.get('count', 0)
        inventory_list = listing.get('inventoryTypes', [])
        if 'resale' in inventory_list[0] if inventory_list else '':
            resale_min.append(min_price)
            resale_max.append(max_price)
            resale_count += count
        else:
            standard_min.append(min_price)
            standard_max.append(max_price)
            standard_count += count

    return average_of_list(standard_min), average_of_list(standard_max), average_of_list(resale_min), average_of_list(
        resale_max), standard_count, resale_count

### Tickets Parsing ####

def _norm_section(s: str) -> str:
    return (s or "").strip().upper().replace(" ", "")

def _norm_row(s: str) -> str:
    return (s or "").strip().upper()

def _norm_seat(s: str) -> str:
    return str(s).strip().upper()

def build_ticket_id(event_id: str, section: str, row: str, seat: str) -> str:
    # Matches your PK scheme: EventID + '_' + Section + '_' + Row + '_' + Seat
    return f"{event_id}_{_norm_section(section)}_{_norm_row(row)}_{_norm_seat(seat)}"

def parse_offers_to_rows(timestamp_uuid: str, event_id: str, offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert the 'offers' API (list[dict]) into per-seat rows for DB upserts.

    Each returned dict has:
      - EventID, Section, Row, Seat, TicketID
      - TicketPrice (uses faceValue)
      - Currency
      - SellableQuantitiesJson (JSON string of list[int])

    Rules:
      - If any of section/row/seatFrom/seatTo missing -> skip that offer.
      - seatFrom..seatTo is inclusive; emit one row per seat.
      - If seatFrom/seatTo not parseable as ints -> skip that offer.
    """
    rows: List[Dict[str, Any]] = []
    if not offers:
        return rows

    for offer in offers:
        section = offer.get("section")
        row = offer.get("row")
        seat_from = offer.get("seatFrom")
        seat_to = offer.get("seatTo")

        # Must have all four keys
        if not (section and row and seat_from is not None and seat_to is not None):
            continue

        # Coerce seat range to ints
        try:
            start = int(str(seat_from).strip())
            end = int(str(seat_to).strip())
        except Exception:
            continue

        if end < start:
            # If ever reversed, swap (or skip; here we swap to be forgiving)
            start, end = end, start

        # Price & other fields
        price = offer.get("faceValue")  # per requirement, use faceValue
        currency = offer.get("currency", "USD")
        sellable_quantities = offer.get("sellableQuantities") or []

        # Ensure JSON text for DB column
        sellable_json = json.dumps(sellable_quantities)

        # Emit one row per seat in the range (inclusive)
        for seat_num in range(start, end + 1):
            seat_str = str(seat_num)
            ticket_id = build_ticket_id(event_id, section, row, seat_str)

            rows.append({
                "TicketPriceID": generate_timestamped_uuid(),
                "TimestampUUID": timestamp_uuid,
                "EventID": event_id,
                "Section": section,
                "Row": row,
                "Seat": seat_str,
                "TicketID": ticket_id,
                "TicketPrice": price,
                "Currency": currency,
                "SellableQuantitiesJson": sellable_json,
            })

    return rows

### app.log parsing ###

def parse_facets_list(blob: str):
    """
    Extracts and returns the Python list that follows 'Facets:' in a log blob.
    Returns the parsed list object, or None if not found / malformed.

    Handles extra text after the list (e.g., ', _links: {...}') by stopping at the
    matching closing ']' for the opening '[' of the facets list.
    """
    key = "Facets:"
    i = blob.find(key)
    if i == -1:
        return None

    # Start scanning right after 'Facets:' and any whitespace
    s = blob[i + len(key):].lstrip()
    start = s.find('[')
    if start == -1:
        return None

    # Find the matching closing bracket for that opening '['
    depth = 0
    in_str = False
    quote = None
    escape = False
    end = None

    for j, ch in enumerate(s[start:], start=start):
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = True
                quote = ch
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break

    if end is None:
        # Unbalanced brackets
        return None

    list_text = s[start:end]
    # Now it's a clean Python literal list: [{'...': ...}, ...]
    return ast.literal_eval(list_text)

def find_event_results(log_path: Union[str, Path], event_id: str) -> Tuple[bool, Optional[List[Optional[float]]]]:
    """
    Search `app.log` for an event ID.

    - If the event ID is found, return (True, results).
    - If a `Results:` line appears after the event ID, parse it into a list of floats or None.
    - If no `Results:` line appears and another "Running" block begins first, assume no results exist and return (True, None).
    - If the event ID never appears, return (False, None).

    Parameters
    ----------
    log_path : str | Path
        Path to the app.log file.
    event_id : str
        Event ID to search for.

    Returns
    -------
    (found: bool, results: Optional[List[Optional[float]]])
    """
    log_path = Path(log_path)
    found = False
    count_inserts = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "elements" in line:
                count_inserts += 1
            # Detect start of event block
            if not found and event_id in line:
                found = True
                continue

            if found:
                # If we encounter a new "Running" block before results → no results exist
                if "Running" in line and event_id not in line:
                    return True, None

                # if "Data:" in line and "block" in line and event_id in line:
                #     return False, None
                #
                # if "Error processing request for id" and event_id in line:
                #     return False, None

                if "Facets:" in line and event_id in line:
                    try:

                        # Safely evaluate to a Python object
                        facets = parse_facets_list(line)
                        stan_min, stan_max, resl_min, resl_max, stan_count, resl_count = parse_facet(facets)
                        results = [stan_min, stan_max, resl_min, resl_max, stan_count, resl_count]
                    except Exception as e:
                        logger.error(f"Failed tp parse facets: {e}")
                        return True, None

                    return True, results
    # Reached EOF
    return found, None