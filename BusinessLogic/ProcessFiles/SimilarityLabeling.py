import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class SimilarityLabeling:
    def __init__(self, network_similarities):
        self.network_similarities = network_similarities

    def __getThresholdBasedOnClustering(self, similarity_type):
        similarity_measures_df = self.network_similarities.getSimilarityMeasures()

        scores = similarity_measures_df[similarity_type].values.reshape(-1, 1)

        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(scores)

        cluster_centers = kmeans.cluster_centers_.flatten()
        threshold = np.mean(cluster_centers)

        silhouette_avg = silhouette_score(scores, clusters)

        print(f"Threshold based on clustering: {threshold:.2f}")
        print(f"Silhouette Score: {silhouette_avg:.2f}")
        print(f"Percentile {np.percentile(similarity_measures_df[similarity_type], 90)}")

        # TODO: maybe saving threshold somewhere
        return threshold

    def labelSimilarity(self, similarity_type):
        similarity_measures_df = self.network_similarities.getSimilarityMeasures()

        # Check if similarity type is 'Hellinger' or 'NetSimile'
        if similarity_type in ['Hellinger', 'NetSimile']:
            # Invert the comparison: True for values below the threshold, False for values above
            similarity_measures_df["Label " + similarity_type] = similarity_measures_df[
                                                                     similarity_type] < self.__getThresholdBasedOnClustering(
                similarity_type)
        else:
            # Default behavior: True for values above the threshold
            similarity_measures_df["Label " + similarity_type] = similarity_measures_df[
                                                                     similarity_type] >= self.__getThresholdBasedOnClustering(
                similarity_type)

        return self.network_similarities
