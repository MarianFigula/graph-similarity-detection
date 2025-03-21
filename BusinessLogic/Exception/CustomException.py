"""

:exception:

- maly dataset (musia byt aspon 2 grafy v jednom subore na porovnanie alebo jeden graf v jednom a jeden graf v druhom)
- v directory nie je spravny format grafu - ak bude prazdny zoznam tak vtedy
- enabled vahy musia davat sucet 1


"""


class CustomException(Exception):
    def __init__(self, message="Something failed, please try again"):
        self.message = message
        super().__init__(self.message)
