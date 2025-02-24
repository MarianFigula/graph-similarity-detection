from itertools import combinations
import numpy as np
import pandas as pd
from BusinessLogic.ImageModels.ResNetModel import ResNetModel
from BusinessLogic.NetSimileCustom.NetSimileCustom import NetSimileCustom
from BusinessLogic.DataNormaliser.DataNormaliser import DataNormaliser


class NetworkSimilarities:
    def __init__(self, orbit_counts_df):
        self.orbit_counts_df = orbit_counts_df
        self.similarity_measures_df = pd.DataFrame()
        self.orbit_counts_percentage_normalisation = DataNormaliser(
            orbit_counts_df
        ).percentage_normalisation()
        self.column_combinations = sorted(
            list(combinations(sorted(self.orbit_counts_df.columns), 2))
        )

    def getSimilarityMeasures(self):
        return self.similarity_measures_df

    def setSimilarityMeasures(self, similarity_measures_df):
        self.similarity_measures_df = similarity_measures_df

    def computeHellingerSimilarity(self):
        print("computing Hellinger")
        print(self.orbit_counts_df)

        # Compute Hellinger transformations
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

        # Merge with the main similarity DataFrame
        if self.similarity_measures_df.empty:
            self.similarity_measures_df = hellinger_df
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                hellinger_df,
                on=["Graph1", "Graph2"],
                how="outer"
            )

        print("done")

    def computeNetSimileSimilarity(self, path_of_graphs):
        print("computing NetSimile")
        net_simile_results = NetSimileCustom(path_of_graphs).compile_process()

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = net_simile_results
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                net_simile_results,
                on=['Graph1', 'Graph2'],
                how='outer'
            )

        print("done")
    def computeResNetSimilarity(self, img_dir):
        print("computing ResNet")
        resnetModel = ResNetModel(img_dir)

        # Compute similarity
        resnet_similarity = resnetModel.computeSimilarity()

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = resnet_similarity
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                resnet_similarity,
                on=['Graph1', 'Graph2'],
                how='outer'  # Keep all pairs, even if one model doesn't have a score
            )

        print("done")
