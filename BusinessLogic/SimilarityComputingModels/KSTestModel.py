import pandas as pd
from scipy.stats import ks_2samp
from BusinessLogic.DataNormaliser.DataNormaliser import DataNormaliser


class KSTestModel:

    def __init__(self, orbit_counts_df):
        self.orbit_counts_percentage_normalisation = DataNormaliser(
            orbit_counts_df
        ).percentage_normalisation()
        self.alpha = 0.05

    def compute_ks_test_similarity(self):
        graph_names = self.orbit_counts_percentage_normalisation.columns

        similarity_list = []
        ktest_p_values = []
        ktest_labels = []

        for i, graph1 in enumerate(graph_names):
            for j, graph2 in enumerate(graph_names):
                if i >= j:
                    continue

                ks_stat, p_value = ks_2samp(self.orbit_counts_percentage_normalisation[graph1],
                                            self.orbit_counts_percentage_normalisation[graph2])

                label = p_value > self.alpha

                similarity_list.append({
                    "Graph1": graph1,
                    "Graph2": graph2,
                    "KSTest": p_value,
                    # "KSTestLabelAlpha": ktest_labels
                })

        ks_df = pd.DataFrame(similarity_list)

        return ks_df
