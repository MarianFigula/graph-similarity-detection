import pandas as pd
from scipy.stats import ks_2samp


class KSTestModel:

    def __init__(self, orbit_counts_df):
        self.orbit_counts_df = orbit_counts_df
        self.num_graphlet = 2
        self.alpha = 0.05


    def computeKSTestSimilarity(self):
        graph_names = self.orbit_counts_df.columns

        similarity_list = []
        ktest_p_values = []
        ktest_labels = []

        # Porovnanie každého grafu s každým
        for i, graph1 in enumerate(graph_names):
            for j, graph2 in enumerate(graph_names):
                if i >= j:
                    continue  # Vyhneme sa redundantným porovnaniam

                # Zoberieme len graflet G2 (riadok index 1) a vypočítame rozdiel medzi grafmi
                diff_g2 = self.orbit_counts_df.loc[self.num_graphlet, graph2] - self.orbit_counts_df.loc[self.num_graphlet, graph1]  # G2 rozdiel

                # Odpočítame tento rozdiel od všetkých grafletov v graph2
                adjusted_graph = self.orbit_counts_df[graph2] - diff_g2

                # KS test medzi pôvodným a upraveným grafom
                ks_stat, p_value = ks_2samp(self.orbit_counts_df[graph1], adjusted_graph)

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