import os
from itertools import combinations

import numpy as np
import pandas as pd

from training_neural_network.NeuralNetworkModelUtils import NeuralNetworkModelUtils
from training_neural_network.PredictionViewer.PredictionViewer import PredictionViewer


class NeuralNetworkPredictor:
    def __init__(self):
        self.mlp = None
        self.scaler = None
        self.model_dir = '../training_neural_network/saved_models'
        self.output_dir = '../predictions'
        self.modelUtils = NeuralNetworkModelUtils(self.model_dir)
        self.predictionViewer = None

    def load_model(self, option_model):
        self.mlp, self.scaler = self.modelUtils.load_model(option_model)

    def __predict_new_pairs(self, mlp, scaler, graphlet_df, new_pairs):
        """
        Predict similarity for new graph pairs

        Parameters:
        -----------
        mlp : trained MLPClassifier model
        scaler : trained StandardScaler
        graphlet_df : DataFrame containing graphlet distributions for all graphs
        new_pairs : list of tuples [(graph1_id, graph2_id), ...]

        Returns:
        --------
        results : DataFrame with graph pairs and predictions
        """
        results = []

        for graph1_id, graph2_id in new_pairs:
            # Check if both graphs exist in the graphlet dataframe
            if graph1_id in graphlet_df.columns and graph2_id in graphlet_df.columns:
                # Get features for both graphs
                graph1_feat = graphlet_df[graph1_id].values
                graph2_feat = graphlet_df[graph2_id].values

                # Create pair features (same as during training)
                diff_feat = np.abs(graph1_feat - graph2_feat)
                prod_feat = graph1_feat * graph2_feat
                pair_feat = np.concatenate([diff_feat, prod_feat]).reshape(1, -1)

                # Scale features
                pair_feat_scaled = scaler.transform(pair_feat)

                # Make prediction
                pred = mlp.predict(pair_feat_scaled)[0]
                pred_prob = mlp.predict_proba(pair_feat_scaled)[0][1]  # Probability of class 1 (similar)

                results.append({
                    'Graph1': graph1_id,
                    'Graph2': graph2_id,
                    'Prediction': 'Similar' if pred == 1 else 'Not Similar',
                    'Similarity_Score': pred_prob
                })
            else:
                missing = []
                if graph1_id not in graphlet_df.columns:
                    missing.append(graph1_id)
                if graph2_id not in graphlet_df.columns:
                    missing.append(graph2_id)
                print(f"Warning: Could not find graphlet data for {', '.join(missing)}")

        return pd.DataFrame(results)

    def predict(self, graphlet_df, option_model=None):
        self.mlp, self.scaler = self.modelUtils.load_model(option_model)

        graphlet_df = graphlet_df.drop('Unnamed: 0', axis=1)
        graph_names = graphlet_df.columns.tolist()

        new_pairs = list(combinations(graph_names, 2))

        result_df = self.__predict_new_pairs(self.mlp, self.scaler, graphlet_df, new_pairs)

        return result_df

    def predict_two_graphlet_distributions(self, graphlet_df, graphlet_df_2, option_model=None):
        self.mlp, self.scaler = self.modelUtils.load_model(option_model)

        # Clean up the dataframes
        graphlet_df = graphlet_df.drop('Unnamed: 0', axis=1)
        graphlet_df_2 = graphlet_df_2.drop('Unnamed: 0', axis=1)

        # Get graph names from both dataframes
        graph_names = graphlet_df.columns.tolist()
        graph_names_2 = graphlet_df_2.columns.tolist()

        # Create a combined dataframe for prediction
        combined_df = pd.concat([graphlet_df, graphlet_df_2], axis=1)

        # Create pairs between the two sets of graph names
        new_pairs = []
        for graph1 in graph_names:
            for graph2 in graph_names_2:
                new_pairs.append((graph1, graph2))

        # Pass the combined dataframe and pairs to the prediction function
        result_df = self.__predict_new_pairs(self.mlp, self.scaler, combined_df, new_pairs)

        return result_df

    def display_predictions(self, result_df):
        self.predictionViewer = PredictionViewer(result_df)
        self.predictionViewer.showPredictions()

    def download_predictions(self, result_df):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_file = os.path.join(self.output_dir, 'predictions.csv')
        result_df.to_csv(output_file, index=False)
