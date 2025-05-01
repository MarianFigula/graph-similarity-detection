from BusinessLogic.Exception.WeightSumException import WeightSumException
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

        if netsimile_check_val:
            self.network_similarities.computeNetSimileSimilarity(self.path)

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
                          ks_check_val=False,
                          hellinger_weight=0.0,
                          netsimile_weight=0.0,
                          resnet_weight=0.0,
                          ks_weight=0.0):

        total_weight = 0.0
        if hellinger_check_val:
            total_weight += hellinger_weight
        if netsimile_check_val:
            total_weight += netsimile_weight
        if resnet_check_val:
            total_weight += resnet_weight
        if ks_check_val:
            total_weight += ks_weight

        if total_weight != 1.0:
            raise WeightSumException(f"The sum of weights must be 1.0. Current sum: {total_weight}")

        self.similarity_label_handler = SimilarityLabeling(self.network_similarities)

        method_labels = {}


        if hellinger_check_val:
            method_labels["Hellinger"] = self.similarity_label_handler.labelSimilarity("Hellinger", hellinger_weight)

        if netsimile_check_val:
            method_labels["NetSimile"] = self.similarity_label_handler.labelSimilarity("NetSimile", netsimile_weight)

        if resnet_check_val:
            method_labels["ResNet"] = self.similarity_label_handler.labelSimilarity("ResNet", resnet_weight)

        if ks_check_val:
            method_labels["KSTest"] = self.similarity_label_handler.labelSimilarity("KSTest", ks_weight)

        # Combine weighted results if there's more than one method enabled
        if len(method_labels) > 0:
            self.similarity_label_handler.combineWeightedLabels(method_labels)

        return self.network_similarities