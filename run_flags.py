"""
This python file can be used to configure what is run when runner.py file is run
"""

# Delete db toggle
DELETE = False
# Create tables toggle
CREATE = False
# Backup DB after event insert Toggle
BACKUP = False
# Run API toggle
RUN = True if not(DELETE or CREATE or BACKUP) else False
# Test Mode toggle (reduces instances retrieved from API), uses testdb
TEST = False

### Scrapes all available tickets for each event
SCRAPE_TICKETS = True

### Run ML process
ML_RUN = True


