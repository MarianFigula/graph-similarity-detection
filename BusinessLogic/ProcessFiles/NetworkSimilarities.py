import pandas as pd
from BusinessLogic.SimilarityComputingModels.HellingerDistanceModel import HellingerDistanceModel
from BusinessLogic.SimilarityComputingModels.KSTestModel import KSTestModel
from BusinessLogic.SimilarityComputingModels.ResNetModel import ResNetModel
from BusinessLogic.SimilarityComputingModels.NetSimileModel import NetSimileModel


class NetworkSimilarities:
    def __init__(self, orbit_counts_df):
        self.orbit_counts_df = orbit_counts_df
        self.similarity_measures_df = pd.DataFrame()

    def getSimilarityMeasures(self, filterTheSameGraphs=False):
        if filterTheSameGraphs:
            self.similarity_measures_df['Type1'] = self.similarity_measures_df['Graph1'].apply(
                lambda x: x.split('_')[0])
            self.similarity_measures_df['Type2'] = self.similarity_measures_df['Graph2'].apply(
                lambda x: x.split('_')[0])
            self.similarity_measures_df = self.similarity_measures_df[
                self.similarity_measures_df['Type1'] != self.similarity_measures_df['Type2']]
            self.similarity_measures_df = self.similarity_measures_df.drop(columns=['Type1', 'Type2'])

        return self.similarity_measures_df

    def setSimilarityMeasures(self, similarity_measures_df):
        self.similarity_measures_df = similarity_measures_df

    def computeHellingerSimilarity(self):

        hellingerSimilarityModel = HellingerDistanceModel(self.orbit_counts_df)

        hellingerSimilarityResults = hellingerSimilarityModel.computeHellingerDistance()

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = hellingerSimilarityResults
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                hellingerSimilarityResults,
                on=["Graph1", "Graph2"],
                how="inner"
            )

        print("done")

    def computeNetSimileSimilarity(self, path_of_graphs):
        print("computing NetSimile")

        netSimileModel = NetSimileModel(path_of_graphs)

        netSimileResults = netSimileModel.compile_process()

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = netSimileResults
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                netSimileResults,
                on=['Graph1', 'Graph2'],
                how='inner'
            )

        print("done")

    def computeResNetSimilarity(self, img_dir):
        print("computing ResNet")
        resnetModel = ResNetModel(img_dir)

        resnet_similarity = resnetModel.computeSimilarity()

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

    def computeKSTestSimilarity(self):
        ksTestModel = KSTestModel(self.orbit_counts_df)

        ks_test_similarity_df = ksTestModel.computeKSTestSimilarity()

        if self.similarity_measures_df.empty:
            self.similarity_measures_df = ks_test_similarity_df
        else:
            self.similarity_measures_df = pd.merge(
                self.similarity_measures_df,
                ks_test_similarity_df,
                on=['Graph1', 'Graph2'],
                how='inner'
            )
