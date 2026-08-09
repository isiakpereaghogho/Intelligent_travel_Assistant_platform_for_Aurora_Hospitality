import sys
import logging

def error_message_details(error, error_detail: sys):

    #extract traceback details to get file name and line number where the error occurred
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename # Get the file name where the error occurred

    # create a detailed error message with file name, line number and error message
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name} at line: {line_number} with error message: {str(error)}"
    
    #log the error message using the logging module
    logging.error(error_message)

    return error_message

class CustomException(Exception):   
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message) # Call the base class constructor with the error message
        self.error_message = error_message_details(error_message, error_detail)# Get the detailed error message using the error_message_details function

    def __str__(self) -> str:
        # Return the detailed error message when the exception is converted to a string
        return self.error_message