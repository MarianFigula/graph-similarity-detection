from functools import partial

from BusinessLogic.ProcessFiles.NetworkSimilarities import NetworkSimilarities


class SimilarityCalculation:
    def __init__(self, orbit_counts_df, path, img_dir):
        self.network_similarities = NetworkSimilarities(orbit_counts_df)
        self.path = path
        self.img_dir = img_dir

    def countSimilarities(
            self,
            hellinger_check_val=False,
            netsimile_check_val=False,
            resnet_check_val=False):

        if hellinger_check_val:
            self.network_similarities.computeHellingerSimilarity()

        if netsimile_check_val:
            self.network_similarities.computeNetSimileSimilarity(self.path)

        if resnet_check_val:
            self.network_similarities.computeResNetSimilarity(self.img_dir)

        return self.network_similarities

    def exportSimilarity(self, path_to_export):
        self.network_similarities.exportSimilarityMeasures(path_to_export)

