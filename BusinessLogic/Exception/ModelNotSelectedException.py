from BusinessLogic.Exception.CustomException import CustomException


class ModelNotSelectedException(CustomException):
    def __init__(self, message="Model not selected, please try again"):
        super().__init__(message)