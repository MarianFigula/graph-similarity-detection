import uuid

import optuna
from optuna.samplers import TPESampler
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter

from BusinessLogic.DataNormaliser.DataNormaliser import DataNormaliser


def load_data(embeddings_path, similarities_path):
    """
    Load graph embeddings and similarity data.
    """
    # Load embeddings
    embeddings_df = pd.read_csv(embeddings_path)
    embeddings_df = DataNormaliser(embeddings_df).log_scale_percentage_normalisation()

    # Transpose the dataframe
    embeddings_df_transposed = embeddings_df.transpose()
    embeddings_df_transposed.columns = embeddings_df_transposed.iloc[0]
    embeddings_df_transposed = embeddings_df_transposed.drop(embeddings_df_transposed.index[0])

    # Create dictionary
    embeddings = {graph_name: embeddings_df_transposed.loc[graph_name].values.astype(float)
                  for graph_name in embeddings_df_transposed.index}

    # Load similarities
    similarities = pd.read_csv(similarities_path)
    similarities = similarities.sample(frac=1, random_state=42).reset_index(drop=True)
    graph_names = (similarities['Graph1'], similarities['Graph2'])

    return embeddings, similarities, graph_names


def prepare_training_data(embeddings, similarities):
    """
    Prepare training data for Siamese network.
    """
    pairs = []
    labels = []

    for _, row in similarities.iterrows():
        graph1_id = row['Graph1']
        graph2_id = row['Graph2']
        is_similar = 1 if row['Label'] else 0

        # Get embeddings for the pair
        emb1 = embeddings[graph1_id]
        emb2 = embeddings[graph2_id]

        pairs.append([emb1, emb2])
        labels.append(is_similar)

    # Convert to numpy arrays
    pairs = np.array(pairs, dtype=object)
    labels = np.array(labels)

    return pairs, labels


def create_base_network(input_dim, hidden_units, num_hidden_layers, activation, dropout_rate):
    """
    Create the base network for the Siamese architecture with configurable hyperparameters.
    """
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    x = input_layer

    # Add hidden layers
    for i in range(num_hidden_layers):
        x = tf.keras.layers.Dense(hidden_units, activation=activation)(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    output_layer = tf.keras.layers.Dense(30)(x)  # Fixed output size for embedding

    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def create_siamese_network(input_dim, hidden_units, num_hidden_layers, activation, dropout_rate, learning_rate):
    """
    Create the Siamese network with configurable hyperparameters.
    """
    # Define the base network
    base_network = create_base_network(input_dim, hidden_units, num_hidden_layers, activation, dropout_rate)

    # Define the inputs for the two branches
    input_a = tf.keras.layers.Input(shape=(input_dim,))
    input_b = tf.keras.layers.Input(shape=(input_dim,))

    # Get embeddings for both inputs using the same base network
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Compute the L1 distance between the embeddings
    distance = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([processed_a, processed_b])

    # Add a dense layer for final prediction
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(distance)

    # Create the model
    model = tf.keras.Model(inputs=[input_a, input_b], outputs=prediction)

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    return model


def objective(trial):
    """
    Objective function for Optuna to optimize.
    """
    # Define hyperparameters to optimize
    hidden_units = trial.suggest_int('hidden_units', 16, 64, step=16)
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid'])
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.006, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 10, 50)

    # Load and prepare data (only once per study if possible, here we do it for each trial)
    embeddings_path = "../train_data/graphlet_counts_train.csv"
    similarities_path = "../train_data/similarities_train.csv"
    embeddings, similarities, _ = load_data(embeddings_path, similarities_path)

    # Determine embedding dimension
    input_dim = len(next(iter(embeddings.values())))

    # Prepare training data
    pairs, labels = prepare_training_data(embeddings, similarities)

    # Split data into pairs
    X1 = np.array([pair[0] for pair in pairs])
    X2 = np.array([pair[1] for pair in pairs])

    # Scale the features
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.transform(X2)  # Use the same scaler

    # Split into training, validation, and test sets
    X1_train, X1_temp, X2_train, X2_temp, y_train, y_temp = train_test_split(
        X1, X2, labels, test_size=0.2, random_state=42)

    X1_val, X1_test, X2_val, X2_test, y_val, y_test = train_test_split(
        X1_temp, X2_temp, y_temp, test_size=0.5, random_state=42)

    # Create the model with the trial hyperparameters
    model = create_siamese_network(
        input_dim=input_dim,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        activation=activation,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    # Define callbacks for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        [X1_train, X2_train],
        y_train,
        validation_data=([X1_val, X2_val], y_val),
        batch_size=batch_size,
        epochs=epochs,  # Set high, early stopping will handle it
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate([X1_val, X2_val], y_val, verbose=0)

    # Report intermediate values to Optuna for pruning
    trial.report(val_accuracy, step=1)

    # Handle pruning based on the intermediate value
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_accuracy


def run_optimization(n_trials=50):
    """
    Run the hyperparameter optimization.
    """
    study = optuna.create_study(
        direction='maximize',  # We want to maximize validation accuracy
        sampler=TPESampler(seed=42),  # For reproducibility
        pruner=optuna.pruners.MedianPruner()  # Prune unpromising trials
    )

    study.optimize(objective, n_trials=n_trials)

    # Print optimization results
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig('optuna_optimization_history.png')

    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig('optuna_param_importances.png')

    # Plot parallel coordinate
    plt.figure(figsize=(15, 8))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()
    plt.savefig('optuna_parallel_coordinate.png')

    return study


def train_best_model(study):
    """
    Train a model with the best hyperparameters found by Optuna.
    """
    # Load and prepare data 
    embeddings_path = "../train_data/graphlet_counts_train.csv"
    similarities_path = "../train_data/similarities_train.csv"
    embeddings, similarities, graph_names = load_data(embeddings_path, similarities_path)

    # Determine embedding dimension
    input_dim = len(next(iter(embeddings.values())))

    # Prepare training data
    pairs, labels = prepare_training_data(embeddings, similarities)

    # Split data into pairs
    X1 = np.array([pair[0] for pair in pairs])
    X2 = np.array([pair[1] for pair in pairs])

    # Scale the features
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.transform(X2)  # Use the same scaler

    # Split into training, validation, and test sets
    X1_train, X1_temp, X2_train, X2_temp, y_train, y_temp = train_test_split(
        X1, X2, labels, test_size=0.3, random_state=42)

    X1_val, X1_test, X2_val, X2_test, y_val, y_test = train_test_split(
        X1_temp, X2_temp, y_temp, test_size=0.5, random_state=42)

    # Get best hyperparameters
    best_params = study.best_trial.params

    # Create the model with the best hyperparameters
    best_model = create_siamese_network(
        input_dim=input_dim,
        hidden_units=best_params['hidden_units'],
        num_hidden_layers=best_params['num_hidden_layers'],
        activation=best_params['activation'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    u = uuid.uuid4().int
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'best_siamese_model_optuna_{u}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train the model
    history = best_model.fit(
        [X1_train, X2_train],
        y_train,
        validation_data=([X1_val, X2_val], y_val),
        batch_size=best_params['batch_size'],
        epochs=best_params['epochs'],  # We can train longer with early stopping
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('best_model_training_history.png')

    # Evaluate on test set
    test_loss, test_accuracy = best_model.evaluate([X1_test, X2_test], y_test)
    print(f"\nTest accuracy with best hyperparameters: {test_accuracy:.4f}")

    # Save the final model
    best_model.save("siamese_graph_model_optimized")

    # Save the scaler for preprocessing new data
    import joblib
    joblib.dump(scaler, 'siamese_graph_scaler.pkl')

    return best_model, scaler, (X1_test, X2_test, y_test)


if __name__ == "__main__":
    print("Starting Siamese Network hyperparameter optimization with Optuna...")

    # Run hyperparameter optimization
    study = run_optimization(n_trials=30)  # Adjust number of trials as needed

    # Train model with the best hyperparameters
    best_model, scaler, test_data = train_best_model(study)

    # Visualize final model performance
    X1_test, X2_test, y_test = test_data
    y_pred = (best_model.predict([X1_test, X2_test]) >= 0.5).astype(int).flatten()

    # Show performance metrics
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Best Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('best_model_confusion_matrix.png')

    print("\nOptimization and evaluation complete!")
    print(f"Best model saved as 'siamese_graph_model_optimized'")
    print(f"Scaler saved as 'siamese_graph_scaler.pkl'")