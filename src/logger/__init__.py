import logging
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Logging configuration/constants
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 5 # Number of backup log files to keep

# Create log directory if it doesn't exist
BASE_DIR = Path(__file__).resolve().parent.parent.parent
log_dir_path = BASE_DIR / LOG_DIR
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = log_dir_path / LOG_FILE

def setup_logger():
    # Create a logger with a rotating file handler and console handler
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    #Prevent duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger
    
    # Define log message format(formatter)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_FILE_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Log all levels to file  

    # Console handler with UTF-8 encoding and error handling
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # Log INFO and above to console

    # Add custon filter to handle unicode errors
    class UnicodeSafeFilter(logging.Filter):
        def filter(self, record):
            if record.msg:
                #Replace problematic character
                if isinstance(record.msg, str):
                    #Replace common problematic unicode character
                    record.msg = record.msg.replace('\u019f', 'o')
                    record.msg = record.msg.replace('\ufb00', 'ff')
                    record.msg = record.msg.replace('\u2011', '-')
                    record.msg = record.msg.replace('\u2013', '-')
                    record.msg = record.msg.replace('\u2014', '-')
                    record.msg = record.msg.replace('\uf0b7', '*')
                    record.msg = record.msg.encode('ascii', 'ignore').decode('ascii')
            return True

    console_handler.addFilter(UnicodeSafeFilter())
    file_handler.addFilter(UnicodeSafeFilter())

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

