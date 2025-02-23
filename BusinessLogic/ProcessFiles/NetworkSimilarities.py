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

    def computeHellingerSimilarity(self):
        print("computing Hellinger")
        print(self.orbit_counts_df)

        # Compute Hellinger transformations
        computations = self.orbit_counts_percentage_normalisation.apply(
            lambda col: col.map(lambda val: np.sqrt(val))
        )
        computed_columns = []
        pair_names = []

        for col1, col2 in self.column_combinations:
            computed_columns.append((computations[col1] - computations[col2]) ** 2)
            pair_names.append(f"{col1}---{col2}")

        result_df = pd.concat(computed_columns, axis=1)
        result_df.columns = pair_names

        hellinger_scores = np.sqrt(result_df.sum()) / np.sqrt(2)

        hellinger_df = pd.DataFrame({
            "Pair": pair_names,
            "Hellinger": hellinger_scores.values
        })

        # Merge with the main similarity DataFrame
        if self.similarity_measures_df.empty:
            self.similarity_measures_df = hellinger_df
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                hellinger_df,
                on="Pair",
                how="outer"
            )

        print("done")

    def computeNetSimileSimilarity(self, path_of_graphs):
        print("computing NetSimile")
        net_simile_results = NetSimileCustom(path_of_graphs).compile_process()

        net_simile_results.reset_index(inplace=True)
        net_simile_results.columns = ['Pair'] + list(net_simile_results.columns[1:])

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = net_simile_results
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                net_simile_results,
                on='Pair',
                how='outer'
            )

        print("done")
    def computeResNetSimilarity(self, img_dir):
        print("computing ResNet")
        resnetModel = ResNetModel(img_dir)

        # Compute similarity
        resnet_similarity = resnetModel.computeSimilarity()

        resnet_similarity.reset_index(inplace=True)
        resnet_similarity.columns = ['Pair'] + list(resnet_similarity.columns[1:])

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = resnet_similarity
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                resnet_similarity,
                on='Pair',  # Merge based on the pair names
                how='outer'  # Keep all pairs, even if one model doesn't have a score
            )

        print("done")

    def exportSimilarityMeasures(self, path_to_export):
        self.similarity_measures_df.to_csv(path_to_export)