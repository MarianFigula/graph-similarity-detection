from itertools import combinations

import numpy as np
import pandas as pd

from BusinessLogic.DataNormaliser.DataNormaliser import DataNormaliser


class HellingerDistanceModel:
    def __init__(self, orbit_counts_df):
        self.orbit_counts_df = orbit_counts_df
        self.column_combinations = sorted(
            list(combinations(sorted(self.orbit_counts_df.columns), 2))
        )
        self.orbit_counts_percentage_normalisation = DataNormaliser(
            orbit_counts_df
        ).percentage_normalisation()

    def computeHellingerDistance(self):
        print("computing Hellinger")
        print(self.orbit_counts_df)

        computations = self.orbit_counts_percentage_normalisation.apply(
            lambda col: col.map(lambda val: np.sqrt(val))
        )
        computed_columns = []
        pair_data = []

        for col1, col2 in self.column_combinations:
            computed_columns.append((computations[col1] - computations[col2]) ** 2)
            pair_data.append((col1, col2))

        result_df = pd.concat(computed_columns, axis=1)
        hellinger_scores = np.sqrt(result_df.sum()) / np.sqrt(2)

        hellinger_df = pd.DataFrame({
            "Graph1": [pair[0] for pair in pair_data],
            "Graph2": [pair[1] for pair in pair_data],
            "Hellinger": hellinger_scores.values
        })

        return hellinger_df
