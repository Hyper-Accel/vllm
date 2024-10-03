import pytest
import logging
from datetime import datetime

def pytest_configure(config):
    # Get the current date for dynamic log filename
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"logs/verification_log_{current_date}.log"
    
    # Set up the logging configuration with the dynamically generated log filename
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(levelname)-8s %(asctime)s %(name)s::%(filename)s:%(funcName)s:%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Optionally, you can also print to the console in addition to logging to the file
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)s %(message)s'))
    logging.getLogger().addHandler(console_handler)