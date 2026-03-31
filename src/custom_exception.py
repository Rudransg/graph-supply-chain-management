import sys
from typing import Optional


class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: Optional[Exception] = None):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(
        error_message: str,
        error_detail: Optional[Exception],
    ) -> str:
        # use traceback of the current exception context
        _, _, exc_tb = sys.exc_info()
        if exc_tb is None:
            # fallback: just return message + original exception (if any)
            if error_detail is not None:
                return f"{error_message} - {error_detail}"
            return error_message

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        if error_detail is not None:
            return (
                f"Error in {file_name} , line {line_number} : "
                f"{error_message} - {error_detail}"
            )
        return f"Error in {file_name} , line {line_number} : {error_message}"

    def __str__(self) -> str:
        return self.error_message
