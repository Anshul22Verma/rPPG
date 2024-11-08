import logging
import os

# Automatically set up logging when this module is imported
log_file = "rPPG.log"
log_level = logging.DEBUG

# Ensure the log directory exists
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create the root logger
logger = logging.getLogger()

# Set the log level for the root logger
logger.setLevel(log_level)

# File handler for logging to a file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(log_level)

# Console handler for logging to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Adjust to desired level for console output

# Define the log format
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# For module-specific logs
# logger = logging.getLogger(__name__)
