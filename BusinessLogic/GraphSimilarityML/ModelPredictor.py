import os
from datetime import datetime
from itertools import combinations
import numpy as np
import pandas as pd
from BusinessLogic.Exception.CustomException import CustomException
from BusinessLogic.Exception.EmptyDataException import EmptyDataException
from BusinessLogic.GraphSimilarityML.ModelUtils import ModelUtils
from BusinessLogic.GraphSimilarityML.PredictionViewer.PredictionViewer import PredictionViewer


class ModelPredictor:
    def __init__(self):
        self.model = None
        self.model_dir = 'MachineLearningData/saved_models'
        self.output_dir = 'MachineLearningData/predictions'
        self.model_utils = ModelUtils(self.model_dir)
        self.predictionViewer = None

    def __predict_new_pairs(self, model, graphlet_df, new_pairs, option_model=None):
        """
        Predict similarity for new graph pairs

        Parameters:
        -----------
        mlp : trained RandomForestClassifier model
        graphlet_df : DataFrame containing graphlet distributions for all graphs
        new_pairs : list of tuples [(graph1_id, graph2_id), ...]

        Returns:
        --------
        results : DataFrame with graph pairs and predictions
        """
        results = []

        for graph1_id, graph2_id in new_pairs:
            if graph1_id in graphlet_df.columns and graph2_id in graphlet_df.columns:
                if graph1_id == graph2_id:
                    raise CustomException("Same graph ids in both datasets")

                graph1_feat = graphlet_df[graph1_id].values
                graph2_feat = graphlet_df[graph2_id].values

                pair_feat = np.concatenate([graph1_feat, graph2_feat]).reshape(1, -1)

                if option_model.endswith('.h5'):
                    proba = model.predict(pair_feat)[0, 0]
                    pred = (proba >= 0.5).astype(int)
                else:
                    pred = model.predict(pair_feat)[0]
                    proba = model.predict_proba(pair_feat)[0, 1]

                results.append((graph1_id, graph2_id, pred, proba))
            else:
                missing = []
                if graph1_id not in graphlet_df.columns:
                    missing.append(graph1_id)
                if graph2_id not in graphlet_df.columns:
                    missing.append(graph2_id)
                print(f"Warning: Could not find graphlet data for {', '.join(missing)}")

        return pd.DataFrame(results, columns=['Graph1', 'Graph2', 'Similarity', 'Probability'])

    def predict(self, graphlet_df, option_model=None):
        self.model = self.model_utils.load_model(option_model)

        if 'Unnamed: 0' in graphlet_df.columns:
            graphlet_df = graphlet_df.drop('Unnamed: 0', axis=1)

        graph_names = graphlet_df.columns.tolist()

        if len(graph_names) < 2:
            raise EmptyDataException("Dataframe must contain at least two graphs.")

        new_pairs = list(combinations(graph_names, 2))

        result_df = self.__predict_new_pairs(self.model, graphlet_df, new_pairs, option_model)

        return result_df

    def predict_two_graphlet_distributions(self, graphlet_df, graphlet_df_2, option_model=None):
        self.model = self.model_utils.load_model(option_model)

        if 'Unnamed: 0' in graphlet_df.columns:
            graphlet_df = graphlet_df.drop('Unnamed: 0', axis=1)
        if 'Unnamed: 0' in graphlet_df_2.columns:
            graphlet_df_2 = graphlet_df_2.drop('Unnamed: 0', axis=1)

        graph_names = graphlet_df.columns.tolist()
        graph_names_2 = graphlet_df_2.columns.tolist()

        if len(graph_names) < 1 or len(graph_names_2) < 1:
            raise EmptyDataException("Both dataframes must contain at least one graph.")

        combined_df = pd.concat([graphlet_df, graphlet_df_2], axis=1)

        new_pairs = []
        for graph1 in graph_names:
            for graph2 in graph_names_2:
                new_pairs.append((graph1, graph2))

        result_df = self.__predict_new_pairs(self.model, combined_df, new_pairs, option_model)

        return result_df

    def display_predictions(self, result_df):
        self.predictionViewer = PredictionViewer(result_df)
        self.predictionViewer.show_predictions()

    def download_predictions(self, result_df, option_model=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_file = os.path.join(self.output_dir,
                                   f"predictions_{option_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        result_df.to_csv(output_file, index=False)
        return output_file
