class CustomException(Exception):
    def __init__(self, message="Something failed, please try again"):
        self.message = message
        super().__init__(self.message)
