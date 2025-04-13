import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from training_neural_network import model_utils


# Load the data
def load_data(graphlet_path, similarities_path):
    # Load graphlet counts (features for each graph)
    graphlet_df = pd.read_csv(graphlet_path)

    # Load similarities (pairs of graphs with TRUE/FALSE labels)
    similarities_df = pd.read_csv(similarities_path)
    similarities_df = similarities_df.drop(columns=['Unnamed: 0'])
    similarities_df = similarities_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert boolean labels to integers (0 for FALSE, 1 for TRUE)
    similarities_df['Label'] = similarities_df['Label'].astype(int)

    return graphlet_df, similarities_df


# Create features for each graph pair
def create_pair_features(graphlet_df, similarities_df):
    features = []
    labels = []

    graph_features = {}

    print("Graphlet DataFrame columns:", graphlet_df.columns)
    print("First few rows of graphlet data:")
    print(graphlet_df.head())

    for col in graphlet_df.columns:
        if col != 'Unnamed: 0' and not col.startswith('index'):  # Skip any index columns
            # Each column is a graph, with 30 graphlet counts as features
            graph_id = col
            graph_features[graph_id] = graphlet_df[col].values

    # For each pair of graphs
    for _, row in similarities_df.iterrows():
        graph1_id = row['Graph1']
        graph2_id = row['Graph2']
        label = row['Label']

        # Get features for both graphs
        if graph1_id in graph_features and graph2_id in graph_features:
            graph1_feat = graph_features[graph1_id]
            graph2_feat = graph_features[graph2_id]

            # Create pair features (options: concatenation, absolute difference, element-wise product)
            # Option 1: Concatenation
            # pair_feat = np.concatenate([graph1_feat, graph2_feat])

            # Option 2: Absolute difference (captures similarity)
            diff_feat = np.abs(graph1_feat - graph2_feat)

            # Option 3: Element-wise product (interaction features)
            prod_feat = graph1_feat * graph2_feat

            # Combine different feature types
            pair_feat = np.concatenate([diff_feat, prod_feat])

            features.append(pair_feat)
            labels.append(label)

    return np.array(features), np.array(labels)


# Train the MLP classifier
def train_mlp(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the MLP classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 penalty parameter
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )

    # Train the model
    mlp.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = mlp.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return mlp, scaler, accuracy, report, cm, X_test_scaled, y_test, y_pred


# Plot results
def plot_results(cm, X_test_scaled, y_test, y_pred):
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Similar', 'Similar'],
                yticklabels=['Not Similar', 'Similar'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot decision regions (for 2D visualization)
    if X_test_scaled.shape[1] > 2:
        # Use PCA to reduce to 2D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_test_2d = pca.fit_transform(X_test_scaled)

        plt.figure(figsize=(10, 8))
        plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test,
                    cmap='coolwarm', edgecolors='k')
        plt.title('PCA Projection of Test Data with True Labels')
        plt.colorbar(label='Class')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred,
                    cmap='coolwarm', edgecolors='k')
        plt.title('PCA Projection of Test Data with Predicted Labels')
        plt.colorbar(label='Class')
        plt.show()


# Main function
def main():
    # Paths to data files
    graphlet_path = 'D:/Škola/DP1/project/a_dp/input/all_graphlet_counts_together/graphlet_counts.csv'
    similarities_path = 'D:/Škola/DP1/project/a_dp/input/all_graphlet_counts_together/filtered_similarity_measures.csv'

    print("Loading data...")
    graphlet_df, similarities_df = load_data(graphlet_path, similarities_path)

    print(f"Loaded {len(graphlet_df)} graphs and {len(similarities_df)} pairs")

    print("Creating features for graph pairs...")
    X, y = create_pair_features(graphlet_df, similarities_df)

    print(f"Created features with shape: {X.shape}")
    # print(f"Class distribution: {np.bincount(y)}")

    print("Training MLP classifier...")
    mlp, scaler, accuracy, report, cm, X_test_scaled, y_test, y_pred = train_mlp(X, y)

    print(f"Model accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    print("Plotting results...")
    plot_results(cm, X_test_scaled, y_test, y_pred)

    print("Done!")


    # Save the model and scaler
    model_utils.save_model(mlp, scaler)

    return mlp, scaler


# Function to predict new pairs
def predict_similarity(mlp, scaler, graphlet_df, graph1_id, graph2_id):
    """
    Predict if two graphs are similar based on their graphlet distributions

    Parameters:
    -----------
    mlp : trained MLPClassifier model
    scaler : trained StandardScaler
    graphlet_df : DataFrame containing graphlet distributions
    graph1_id : ID of the first graph
    graph2_id : ID of the second graph

    Returns:
    --------
    prediction : boolean, True if similar, False if not
    probability : float, probability of being similar
    """
    # Assuming the first column in graphlet_df is the graph identifier
    graph_id_col = graphlet_df.columns[0]
    feature_cols = graphlet_df.columns[1:]

    # Get features for both graphs
    graph1_feat = graphlet_df[graphlet_df[graph_id_col] == graph1_id][feature_cols].values[0]
    graph2_feat = graphlet_df[graphlet_df[graph_id_col] == graph2_id][feature_cols].values[0]

    # Create pair features
    diff_feat = np.abs(graph1_feat - graph2_feat)
    prod_feat = graph1_feat * graph2_feat
    pair_feat = np.concatenate([diff_feat, prod_feat]).reshape(1, -1)

    # Scale features
    pair_feat_scaled = scaler.transform(pair_feat)

    # Make prediction
    pred = mlp.predict(pair_feat_scaled)[0]
    pred_prob = mlp.predict_proba(pair_feat_scaled)[0][1]  # Probability of class 1 (similar)

    return bool(pred), pred_prob


# Run if executed as a script
if __name__ == "__main__":
    main()