from itertools import combinations

import pandas as pd
import numpy as np
import pickle
import os


# First, save your trained model and scaler
def save_model(mlp, scaler, output_dir='./saved_models'):
    """
    Save the trained model and scaler

    Parameters:
    -----------
    mlp : trained MLPClassifier model
    scaler : trained StandardScaler
    output_dir : directory to save the model files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model
    with open(os.path.join(output_dir, 'saved_models/graph_similarity.pkl'), 'wb') as f:
        pickle.dump(mlp, f)

    # Save the scaler
    with open(os.path.join(output_dir, 'saved_models/scalers/graph_similarity_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model and scaler saved to {output_dir}")


# Load the saved model and scaler
def load_model(model_dir='.'):
    """
    Load the trained model and scaler

    Parameters:
    -----------
    model_dir : directory where model files are saved

    Returns:
    --------
    mlp : trained MLPClassifier model
    scaler : trained StandardScaler
    """
    # Load the model
    with open(os.path.join(model_dir, 'saved_models/graph_similarity.pkl'), 'rb') as f:
        mlp = pickle.load(f)

    # Load the scaler
    with open(os.path.join(model_dir, 'saved_models/scalers/graph_similarity_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    return mlp, scaler


# Predict for new graph pairs (after training or loading your model)
def predict_new_pairs(mlp, scaler, graphlet_df, new_pairs):
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


# ----------------------
# Example usage scenario
# ----------------------

def main_predict():
    # 1. Either use your existing trained model
    # If you're continuing from a training session:
    # mlp, scaler come from your training code

    # Or load previously saved model
    # mlp, scaler = load_model()

    # 2. Load your graphlet data again
    graphlet_df = pd.read_csv('graphlet_counts.csv')

    # 3. Define new pairs to classify
    # Option 1: Manually specify pairs
    new_pairs = [
        ('ba_graph_2000_1', 'ba_graph_2000_2'),
        ('er_graph_2000_1', 'ws_graph_2000_1'),
        # Add more pairs as needed
    ]

    # Option 2: Generate all possible pairs for a subset of graphs
    selected_graphs = ['ba_graph_2000_1', 'ba_graph_2000_2', 'er_graph_2000_1', 'ws_graph_2000_1']
    all_pairs = [(g1, g2) for i, g1 in enumerate(selected_graphs)
                 for g2 in selected_graphs[i + 1:]]

    # 4. Make predictions
    results = predict_new_pairs(mlp, scaler, graphlet_df, all_pairs)  # or use all_pairs

    # 5. Display and save results
    print("\nPrediction Results:")
    print(results)

    # Save to CSV
    results.to_csv('graph_similarity_predictions.csv', index=False)
    print("Predictions saved to 'graph_similarity_predictions.csv'")

    return results


# Example of how to load from a completely new script
def predict_from_scratch(graphlet_df):
    """
    Example of making predictions in a completely new Python script
    """
    # 1. Load saved model and scaler
    mlp, scaler = load_model()

    # 3. Get all unique graph names
    graphlet_df = graphlet_df.drop('Unnamed: 0', axis=1)
    graph_names = graphlet_df.columns.tolist()

    # 4. Generate all possible pairs of graphs
    new_pairs = list(combinations(graph_names, 2))

    # 4. Make predictions
    results = predict_new_pairs(mlp, scaler, graphlet_df, new_pairs)

    # 5. Display results
    print(results)

    return results


graphlet_df = pd.read_csv("D:/Škola/DP1/project/a_dp/input/group_big_out/graphlet_counts.csv")
graphlet_df_xnetwork = pd.read_csv("D:/Škola/DP1/project/a_dp/input/x_network_graphs/out_files/graphlet_counts.csv")

predict_from_scratch(graphlet_df_xnetwork)