from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import layers, Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

from BusinessLogic.DataNormaliser.DataNormaliser import DataNormaliser


# 1. Load and prepare the data
def load_data(embeddings_path, similarities_path):
    """
    Load graph embeddings and similarity data.

    Parameters:
    - embeddings_path: Path to file containing graphlet counts (embeddings)
      Format: 30 rows of graphlets, with graph names as column headers
    - similarities_path: Path to file containing graph similarities with True/False labels

    Returns:
    - embeddings: Dictionary mapping graph IDs to their embeddings
    - similarities: DataFrame containing pairs of graphs and their similarity labels
    """
    # Load embeddings - transposing to get graphs as rows and graphlet counts as columns
    embeddings_df = pd.read_csv(embeddings_path)
    embeddings_df = DataNormaliser(embeddings_df).log_scale_percentage_normalisation()


    # Transpose the dataframe so graphs become rows and graphlet counts become columns
    embeddings_df_transposed = embeddings_df.transpose()

    # The first row will contain the original row indices which we don't need
    # Set the column names using the first row and then drop it
    embeddings_df_transposed.columns = embeddings_df_transposed.iloc[0]
    embeddings_df_transposed = embeddings_df_transposed.drop(embeddings_df_transposed.index[0])

    # Create dictionary where keys are graph names and values are their graphlet count vectors
    embeddings = {graph_name: embeddings_df_transposed.loc[graph_name].values.astype(float)
                  for graph_name in embeddings_df_transposed.index}

    # Load similarities
    similarities = pd.read_csv(similarities_path)
    similarities = similarities.sample(frac=1, random_state=42).reset_index(drop=True)
    graph_names = (similarities['Graph1'], similarities['Graph2'])

    # print(similarities['Label'].value_counts())
    # return
    return embeddings, similarities, graph_names


def prepare_training_data(embeddings, similarities):
    """
    Prepare training data for Siamese network.

    Parameters:
    - embeddings: Dictionary mapping graph IDs to their embeddings
    - similarities: DataFrame containing pairs of graphs and their similarity labels

    Returns:
    - pairs: List of pairs of embeddings
    - labels: Binary labels (1 for similar, 0 for dissimilar)
    """
    pairs = []
    labels = []

    for _, row in similarities.iterrows():
        graph1_id = row['Graph1']  # Adjust column names as needed
        graph2_id = row['Graph2']
        is_similar = 1 if row['Label'] else 0  # Convert True/False to 1/0

        # Get embeddings for the pair
        emb1 = embeddings[graph1_id]
        emb2 = embeddings[graph2_id]

        pairs.append([emb1, emb2])
        labels.append(is_similar)

    # Convert to numpy arrays
    pairs = np.array(pairs, dtype=object)
    labels = np.array(labels)

    return pairs, labels


# 2. Define the Siamese network
def create_base_network(input_dim):
    """
    Create the base network for the Siamese architecture.

    Parameters:
    - input_dim: Dimension of input embeddings

    Returns:
    - model: Base network
    """
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(input_layer)
    x = layers.Dropout(0.2)(x)
    # x = layers.Dense(15, activation='relu')(x)
    output_layer = layers.Dense(30)(x)

    return Model(inputs=input_layer, outputs=output_layer)


def create_siamese_network(input_dim, units=64, dropout_rate=0.2, learning_rate=0.005):

    """
    Create the Siamese network.

    Parameters:
    - input_dim: Dimension of input embeddings

    Returns:
    - model: Siamese network
    """
    # Define the base network
    base_network = create_base_network(input_dim)

    # Define the inputs for the two branches
    input_a = layers.Input(shape=(input_dim,))
    input_b = layers.Input(shape=(input_dim,))

    # Get embeddings for both inputs using the same base network
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Compute the L1 distance between the embeddings
    distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([processed_a, processed_b])

    prediction = layers.Dense(1, activation='sigmoid')(distance)

    # Create the model
    model = Model(inputs=[input_a, input_b], outputs=prediction)

    return model


# 3. Train the Siamese network
def train_siamese_network(embeddings_path, similarities_path, visualize=True, apply_smote=False):
    """
    Main function to load data and train the Siamese network.

    Parameters:
    - embeddings_path: Path to file containing graphlet counts (embeddings)
    - similarities_path: Path to file containing graph similarities with True/False labels
    - visualize: Whether to visualize training history and predictions

    Returns:
    - trained_model: Trained Siamese network
    """
    # Load data
    embeddings, similarities, graph_names = load_data(embeddings_path, similarities_path)

    # Determine embedding dimension
    input_dim = len(next(iter(embeddings.values())))
    print(input_dim)
    # Prepare training data
    pairs, labels = prepare_training_data(embeddings, similarities)

    # Split data into pairs
    X1 = np.array([pair[0] for pair in pairs])
    X2 = np.array([pair[1] for pair in pairs])


    # Split into training and validation sets
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
        X1, X2, labels, test_size=0.2, random_state=42)

    # Further split test into test and validation
    X1_val, X1_test, X2_val, X2_test, y_val, y_test = train_test_split(
        X1_test, X2_test, y_test, test_size=0.5, random_state=42)

    # scaler_X1 = StandardScaler()
    # X1_train = scaler_X1.fit_transform(X1_train)
    # X1_val = scaler_X1.transform(X1_val)
    # X1_test = scaler_X1.transform(X1_test)
    #
    # scaler_X2 = StandardScaler()
    # X2_train = scaler_X2.fit_transform(X2_train)
    # X2_val = scaler_X2.transform(X2_val)
    # X2_test = scaler_X2.transform(X2_test)

    print("Y Train:\n", pd.Series(y_train).value_counts())

    # Create the Siamese network
    model = create_siamese_network(input_dim)

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.017),
                  metrics=['accuracy'])

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)

    # Model checkpoint to save the best model
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     'best_siamese_model.h5',
    #     monitor='val_accuracy',
    #     save_best_only=True,
    #     mode='max')

    # Apply SMOTE to the combined features
    if apply_smote:
        X_combined = np.hstack((X1_train, X2_train))

        print("Original class distribution:", Counter(y_train))


        smote = SMOTE(random_state=42, sampling_strategy='minority')
        X_combined_res, y_train_res = smote.fit_resample(X_combined, y_train)

        print(f"Resampled data shape: {X_combined_res.shape}")
        print("Resampled class distribution:", Counter(y_train_res))

        # Split the resampled data back into X1 and X2
        X1_train_res = X_combined_res[:, :input_dim]
        X2_train_res = X_combined_res[:, input_dim:]

        # Train the model
        history = model.fit(
            [X1_train_res, X2_train_res], y_train_res,
            validation_data=([X1_val, X2_val], y_val),
            batch_size=32,
            epochs=1,
            callbacks=[early_stopping]
        )
    else:
        # Train the model
        history = model.fit(
            [X1_train, X2_train], y_train,
            validation_data=([X1_val, X2_val], y_val),
            batch_size=32,
            epochs=5,
            callbacks=[early_stopping]
        )


    if visualize:

        # Plot training history
        plot_training_history(history)
        # plot_smote_effect(X_combined, X_combined_res, y_train, y_train_res)

        visualize_predictions(model, X1_test, X2_test, y_test, scaler_X1, scaler_X2, train_data_flag=False)

        # Plot embedding distribution
        # plot_embedding_distribution_3d_interactive(X1, X2, labels, graph_names)

    return model, history, scaler_X1, scaler_X2, (X1_test, X2_test, y_test)


def plot_smote_effect(X_before, X_after, y_before, y_after):
    pca = PCA(n_components=2)
    X_before_2d = pca.fit_transform(X_before)
    X_after_2d = pca.transform(X_after)

    plt.figure(figsize=(12, 5))

    # Before SMOTE
    plt.subplot(1, 2, 1)
    plt.scatter(X_before_2d[:, 0], X_before_2d[:, 1], c=y_before, cmap='coolwarm', alpha=0.6)
    plt.title("Before SMOTE")

    # After SMOTE
    plt.subplot(1, 2, 2)
    plt.scatter(X_after_2d[:, 0], X_after_2d[:, 1], c=y_after, cmap='coolwarm', alpha=0.6)
    plt.title("After SMOTE")

    plt.show()

# Call function to visualize


def plot_training_history(history):
    """
    Plot training & validation accuracy and loss values.

    Parameters:
    - history: History object returned by model.fit()
    """
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    ax1.grid(True)

    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('siamese_training_history.png')
    plt.show()
def visualize_predictions(model, X1_test, X2_test, y_test, scaler1=None, scaler2=None, train_data_flag=False):
    """
    Visualize predictions using confusion matrix and similarity heatmap.

    Parameters:
    - model: Trained Siamese network
    - X1_test, X2_test: Test data pairs
    - y_test: True labels for test data
    - scaler: Optional scaler used during training
    """
    # Make predictions
    if scaler1 is not None and scaler2 is not None:
        X1_test_scaled = scaler1.transform(X1_test)  # Scale X1_test
        X2_test_scaled = scaler2.transform(X2_test)  # Scale X2_test
        y_pred_proba = model.predict([X1_test_scaled, X2_test_scaled])
    else:
        # If no scalers are provided, use the original data
        y_pred_proba = model.predict([X1_test, X2_test])

        # Print predicted probabilities
    print("Predicted probabilities:")
    print(y_pred_proba)

    y_pred = (y_pred_proba >= 0.75).astype(int).flatten()

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix for' + (" Train Data" if train_data_flag else " Test Data"))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Generate similarity heatmap for a subset of test samples
    num_samples = min(50, len(X1_test))  # Limit to 50 samples for visualization

    # similarity_matrix = np.zeros((num_samples, num_samples))
    # for i in range(num_samples):
    #     for j in range(num_samples):
    #         if scaler:
    #             emb1 = scaler.transform(X1_test[i].reshape(1, -1))
    #             emb2 = scaler.transform(X1_test[j].reshape(1, -1))
    #             similarity_matrix[i, j] = model.predict([emb1, emb2])[0][0]
    #         else:
    #             similarity_matrix[i, j] = model.predict([X1_test[i].reshape(1, -1), X1_test[j].reshape(1, -1)])[0][0]
    #
    # # Plot similarity heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(similarity_matrix, cmap='viridis')
    # plt.title('Similarity Heatmap (Test Samples)')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Sample Index')
    # plt.savefig('similarity_heatmap.png')
    # plt.show()


# 4. Evaluate and use the model for predictions
def evaluate_model(model, X1_test, X2_test, y_test):
    """
    Evaluate the trained model on test data.

    Parameters:
    - model: Trained Siamese network
    - X1_test, X2_test: Test data pairs
    - y_test: True labels for test data

    Returns:
    - evaluation metrics
    """
    return model.evaluate([X1_test, X2_test], y_test)


def predict_similarity(model, scaler, embedding1, embedding2, threshold=0.5):
    """
    Predict similarity between two graph embeddings.

    Parameters:
    - model: Trained Siamese network
    - scaler: Fitted StandardScaler
    - embedding1, embedding2: Graph embeddings to compare
    - threshold: Decision threshold (default: 0.5)

    Returns:
    - is_similar: Boolean indicating if graphs are similar
    - similarity_score: Raw similarity score
    """
    # Scale the embeddings
    scaled_emb1 = scaler.transform(embedding1.reshape(1, -1))
    scaled_emb2 = scaler.transform(embedding2.reshape(1, -1))

    # Get prediction
    similarity_score = model.predict([scaled_emb1, scaled_emb2])[0][0]

    # Apply threshold
    is_similar = similarity_score >= threshold

    return is_similar, similarity_score


if __name__ == "__main__":

    embeddings_path = "../train_data/graphlet_counts_train.csv"
    similarities_path = "../train_data/similarities_train.csv"

    # embeddings_path = "../train_data/not_balanced_dataset/graphlet_train.csv"
    # similarities_path = "../train_data/not_balanced_dataset/similarities_train.csv"

    # Train the model with visualization
    model, history, scaler, scaler2, test_data = train_siamese_network(embeddings_path, similarities_path,
                                                              visualize=True, apply_smote=False)

    # Save the model for future use
    model.save("siamese_graph_model")

    # Evaluate the model on test data
    X1_test, X2_test, y_test = test_data
    test_loss, test_accuracy = evaluate_model(model, X1_test, X2_test, y_test)
