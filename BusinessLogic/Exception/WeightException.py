from BusinessLogic.Exception.CustomException import CustomException


class WeightException(CustomException):
    def __init__(self, message="Weight must be between 0 and 1.0"):
        super().__init__(message)
        