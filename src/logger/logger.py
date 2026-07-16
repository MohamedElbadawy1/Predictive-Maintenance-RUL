import logging
import os
from datetime import datetime


class Logger:
    """
    Custom logger that logs messages to both console and a daily log file.
    """

    def __init__(self, logger_name: str = "ML_Project"):
        self.logger = logging.getLogger(logger_name)

        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            return

        self.logger.setLevel(logging.INFO)

        # Create logs directory
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Log file name (daily)
        log_filename = datetime.now().strftime("%Y-%m-%d") + ".log"
        log_path = os.path.join(log_dir, log_filename)

        # Log format
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(filename)s | Line:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger


# Create a global logger instance
logger = Logger().get_logger()