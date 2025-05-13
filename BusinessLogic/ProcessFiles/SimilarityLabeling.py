import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from BusinessLogic.Exception.CustomException import CustomException


class SimilarityLabeling:
    def __init__(self, network_similarities):
        self.network_similarities = network_similarities
        self.thresholds = {}

    def __get_threshold_based_on_clustering(self, similarity_type):
        """
        Calculates the threshold based on clustering for a given similarity type.

        Parameters:
        -----------
        similarity_type : str
            The type of similarity measure (e.g. 'Hellinger', 'NetSimile', etc.).

        Returns:
        --------
        threshold : float
            The calculated threshold value.
        """
        similarity_measures_df = self.network_similarities.get_similarity_measures(filter_the_same_graphs=False)

        if similarity_type not in similarity_measures_df.columns:
            raise CustomException(f"Similarity type '{similarity_type}' not found in similarity measures DataFrame.")


        scores = similarity_measures_df[similarity_type].values.reshape(-1, 1)

        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(scores)

        cluster_centers = kmeans.cluster_centers_.flatten()
        threshold = np.mean(cluster_centers)

        silhouette_avg = silhouette_score(scores, clusters)

        print(f"Threshold based on clustering: {threshold}")
        print(f"Silhouette Score: {silhouette_avg}")
        print(f"Percentile {np.percentile(similarity_measures_df[similarity_type], 90)}")

        self.thresholds[similarity_type] = threshold

        return threshold

    def label_similarity(self, similarity_type, weight=0.0):
        """
        Labels the similarity measures based on the given similarity type and weight.

        Parameters:
        -----------
        similarity_type : str
            The type of similarity measure (e.g. 'Hellinger', 'NetSimile', etc.).
        weight : float, optional
            The weight assigned to the similarity measure (default is 0.0).

        Returns:
        --------
        col_name : str
            The name of the column containing the labeled similarity measures.
        """
        similarity_measures_df = self.network_similarities.get_similarity_measures(filter_the_same_graphs=False)
        threshold = self.__get_threshold_based_on_clustering(similarity_type)

        col_name = f"Label {similarity_type} ({threshold}, w={weight})"

        if similarity_type in ['Hellinger', 'NetSimile']:
            similarity_measures_df[col_name] = similarity_measures_df[similarity_type] < threshold
        else:
            similarity_measures_df[col_name] = similarity_measures_df[similarity_type] >= threshold

        self.network_similarities.similarity_measures_df = similarity_measures_df

        return col_name

    def combine_weighted_labels(self, method_labels):
        """
        Combine labels from different methods using their weights

        Parameters:
        -----------
        method_labels : dict
            Dictionary mapping method names to their label column names
        """
        similarity_measures_df = self.network_similarities.similarity_measures_df

        weights = {}
        for method, col_name in method_labels.items():
            weight_str = col_name.split("w=")[1].rstrip(")")
            weights[col_name] = float(weight_str)

        weighted_score = pd.Series(0, index=similarity_measures_df.index)
        for col_name, weight in weights.items():
            weighted_score += similarity_measures_df[col_name].astype(int) * weight

        similarity_measures_df["Label"] = weighted_score >= 0.5
        similarity_measures_df["Weighted Similarity Score"] = weighted_score

        self.network_similarities.similarity_measures_df = similarity_measures_df