# project/logger.py

import logging


# Set up the logger configuration
def setup_logger():
    logger = logging.getLogger('ticketml.worker')  # You can name your logger

    logger.setLevel(logging.DEBUG)  # Set the default logging level
    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler('ml.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Set a log message format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Call the setup function to create the logger
logger = setup_logger()
