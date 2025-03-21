from BusinessLogic.Exception.CustomException import CustomException


class CannotCompareException(CustomException):
    def __init__(self, message="Cannot compare graphs, please try again"):
        super().__init__(message)
