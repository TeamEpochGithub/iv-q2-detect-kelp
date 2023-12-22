"""Setup the logger."""
import datetime
import logging
import os
# Create a logger
logger = logging.getLogger("logger")
# Set the logging level
logger.setLevel(logging.DEBUG)
# Create a file handler using date and time to create a unique filename
time = datetime.datetime.now().strftime("%d-%m-%Y_%H;%M")
# Create directory if it does not exist
if not os.path.exists("logging/logs"):
    os.makedirs("logging/logs")
file_handler = logging.FileHandler("logging/logs/" + str(time) + ".txt")
# Create a formatter
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s")
# Add the formatter to the file handler
file_handler.setFormatter(formatter)
# Add the file handler to the logger
logger.addHandler(file_handler)
