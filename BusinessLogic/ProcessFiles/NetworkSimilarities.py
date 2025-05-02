import pandas as pd
from BusinessLogic.SimilarityComputingModels.HellingerDistanceModel import HellingerDistanceModel
from BusinessLogic.SimilarityComputingModels.KSTestModel import KSTestModel
from BusinessLogic.SimilarityComputingModels.ResNetModel import ResNetModel
from BusinessLogic.SimilarityComputingModels.NetSimileModel import NetSimileModel


class NetworkSimilarities:
    def __init__(self, orbit_counts_df):
        self.orbit_counts_df = orbit_counts_df
        self.similarity_measures_df = pd.DataFrame()

    def get_similarity_measures(self, filter_the_same_graphs=False):
        if filter_the_same_graphs:
            self.similarity_measures_df['Type1'] = self.similarity_measures_df['Graph1'].apply(
                lambda x: x.split('_')[0])
            self.similarity_measures_df['Type2'] = self.similarity_measures_df['Graph2'].apply(
                lambda x: x.split('_')[0])
            self.similarity_measures_df = self.similarity_measures_df[
                self.similarity_measures_df['Type1'] != self.similarity_measures_df['Type2']]
            self.similarity_measures_df = self.similarity_measures_df.drop(columns=['Type1', 'Type2'])

        return self.similarity_measures_df

    def set_similarity_measures(self, similarity_measures_df):
        self.similarity_measures_df = similarity_measures_df

    def compute_hellinger_similarity(self):

        hellinger_similarity_model = HellingerDistanceModel(self.orbit_counts_df)

        hellinger_similarity_results = hellinger_similarity_model.compute_hellinger_distance()

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = hellinger_similarity_results
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                hellinger_similarity_results,
                on=["Graph1", "Graph2"],
                how="inner"
            )

        print("done")

    def compute_net_simile_similarity(self, path_of_graphs):
        print("computing NetSimile")

        net_simile_model = NetSimileModel(path_of_graphs)

        net_simile_results = net_simile_model.compile_process()

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = net_simile_results
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                net_simile_results,
                on=['Graph1', 'Graph2'],
                how='inner'
            )

        print("done")

    def compute_res_net_similarity(self, img_dir):
        print("computing ResNet")
        resnetModel = ResNetModel(img_dir)

        resnet_similarity = resnetModel.compute_similarity()

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = resnet_similarity
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                resnet_similarity,
                on=['Graph1', 'Graph2'],
                how='inner'
            )

        print("done")

    def compute_ks_test_similarity(self):
        ksTestModel = KSTestModel(self.orbit_counts_df)

        ks_test_similarity_df = ksTestModel.compute_ks_test_similarity()

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = ks_test_similarity_df
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                ks_test_similarity_df,
                on=['Graph1', 'Graph2'],
                how='inner'
            )
