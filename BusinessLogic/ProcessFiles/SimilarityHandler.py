from BusinessLogic.ProcessFiles.NetworkSimilarities import NetworkSimilarities
from BusinessLogic.ProcessFiles.SimilarityLabeling import SimilarityLabeling


class SimilarityHandler:
    def __init__(self, orbit_counts_df, path, img_dir):
        self.similarity_label_handler = None
        self.network_similarities = NetworkSimilarities(orbit_counts_df)
        self.path = path
        self.img_dir = img_dir

    def countSimilarities(
            self,
            hellinger_check_val=False,
            netsimile_check_val=False,
            resnet_check_val=False,
            ks_check_val=False):

        if hellinger_check_val:
            self.network_similarities.computeHellingerSimilarity()

        # "D:/Škola/DP1/project/a_dp/input/group_small_in"
        # "D:/Škola/DP1/project/a_dp/input/konz/in_files"
        #D:/Škola/DP1/project/a_dp/input/x_network_corpus/in_files
        if netsimile_check_val:
            self.network_similarities.computeNetSimileSimilarity("D:/Škola/DP1/project/a_dp/input/konz/in_files") #"D:/Škola/DP1/project/a_dp/input/test"

        if resnet_check_val:
            self.network_similarities.computeResNetSimilarity(self.img_dir)

        if ks_check_val:
            self.network_similarities.computeKSTestSimilarity()

        return self.network_similarities

    def exportSimilarity(self, path_to_export):
        self.network_similarities.getSimilarityMeasures().to_csv(path_to_export)

    def labelSimilarities(self,
                          hellinger_check_val=False,
                          netsimile_check_val=False,
                          resnet_check_val=False,
                          ks_check_val=False):

        self.similarity_label_handler = SimilarityLabeling(self.network_similarities)

        if hellinger_check_val:
            self.similarity_label_handler.labelSimilarity("Hellinger")

        if netsimile_check_val:
            self.similarity_label_handler.labelSimilarity("NetSimile")

        if resnet_check_val:
            self.similarity_label_handler.labelSimilarity("ResNet")

        if ks_check_val:
            self.similarity_label_handler.labelSimilarity("KSTest")

        return self.network_similarities
