import sys
from src.logger.logger import logger


class CustomException(Exception):
    """
    Custom exception that includes:
    - Error message
    - File name
    - Line number
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        _, _, exc_tb = error_detail.exc_info()

        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno

        self.error_message = (
            f"Error occurred in script: [{self.file_name}] "
            f"at line [{self.line_number}] "
            f"with message: [{error_message}]"
        )

        logger.error(self.error_message)

    def __str__(self):
        return self.error_message