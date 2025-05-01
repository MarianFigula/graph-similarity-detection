from BusinessLogic.Exception.CustomException import CustomException


class WeightSumException(CustomException):
    def __init__(self, message="Weight sum of all weights must be 1"):
        super().__init__(message)
        