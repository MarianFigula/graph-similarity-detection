from BusinessLogic.Exception.CustomException import CustomException


class EmptyDataException(CustomException):
    def __init__(self, message="Data is empty"):
        super().__init__(message)
