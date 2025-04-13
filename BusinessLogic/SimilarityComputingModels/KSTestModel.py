import pandas as pd
from scipy.stats import ks_2samp
from BusinessLogic.DataNormaliser.DataNormaliser import DataNormaliser


class KSTestModel:

    def __init__(self, orbit_counts_df):
        self.orbit_counts_percentage_normalisation = DataNormaliser(
            orbit_counts_df
        ).percentage_normalisation()
        self.alpha = 0.05

    # TODO: neodratavat nic od nicoho
    # TODO: zobrat percentulne rozdelenie - prcetage_normalization
    # rozne ale rovnako velke tak moze dat ze su podobne
    # TODO: nerobit log pre vypocet

    # preto sa robi percentualne a nie logaritmicke lebo potom by rozne ale rovnako velke grafy mohli ukazat ze su podobne
    def computeKSTestSimilarity(self):
        graph_names = self.orbit_counts_percentage_normalisation.columns

        similarity_list = []
        ktest_p_values = []
        ktest_labels = []

        # Porovnanie každého grafu s každým
        for i, graph1 in enumerate(graph_names):
            for j, graph2 in enumerate(graph_names):
                if i >= j:
                    continue

                ks_stat, p_value = ks_2samp(self.orbit_counts_percentage_normalisation[graph1],
                                            self.orbit_counts_percentage_normalisation[graph2])

                # Determine similarity label
                label = p_value > self.alpha

                similarity_list.append({
                    "Graph1": graph1,
                    "Graph2": graph2,
                    "KSTest": p_value,
                    # "KSTestLabelAlpha": ktest_labels
                })

        ks_df = pd.DataFrame(similarity_list)

        return ks_df
