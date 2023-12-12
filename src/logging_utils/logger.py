import datetime
import os
import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.DEBUG)

# Create a file handler using date and time to create a unique filename
time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create directory if it does not exist
if not os.path.exists("tracking/logs"):
    os.makedirs("tracking/logs")

file_handler = logging.FileHandler("tracking/logs/" + str(time) + ".txt")

# Create a formatter
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s")

# Add the formatter to the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Log some messages
# logger.debug("A debug message")

# logger.info("An info message")

# logger.warning("A warning message")

# logger.error("An error message")