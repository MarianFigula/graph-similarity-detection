import os
import uuid

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from BusinessLogic.DataNormaliser.DataNormaliser import DataNormaliser
from tensorflow.keras import layers, Model, optimizers, callbacks

class MLPClassifierGraphSimilarity:
    def __init__(self,  graphlet_counts, similarity_measures, hyperparameters):
        self.hyperparameters = hyperparameters
        self.graphlet_counts = graphlet_counts
        self.similarity_measures = similarity_measures

        self.saved_models_dir = 'MachineLearningData/saved_models'
        self.model = None

        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_prob = None
        self.history = None

        self.uuid = uuid.uuid4().int

    def load_data(self):
        if 'Unnamed: 0' in self.graphlet_counts.columns:
            self.graphlet_counts = self.graphlet_counts.drop('Unnamed: 0', axis=1)

        graphlet_df = DataNormaliser(self.graphlet_counts).percentage_normalisation()

        if 'Unnamed: 0' in self.similarity_measures.columns:
            self.similarity_measures = self.similarity_measures.drop('Unnamed: 0', axis=1)

        similarities = self.similarity_measures.sample(frac=1, random_state=42).reset_index(drop=True)

        graphlet_df_transposed = graphlet_df.transpose()
        graphlet_df_transposed.columns = graphlet_df_transposed.iloc[0]
        # embeddings_df_transposed = graphlet_df_transposed.drop(graphlet_df_transposed.index[0])

        embeddings = {graph_name: graphlet_df_transposed.loc[graph_name].values.astype(float)
                      for graph_name in graphlet_df_transposed.index}

        return embeddings, similarities

    def prepare_dataset(self):
        embeddings, similarities = self.load_data()

        X = []
        y = []

        for _, row in similarities.iterrows():
            graph1_id = row['Graph1']
            graph2_id = row['Graph2']
            label = row['Label']

            graph1_features = embeddings[graph1_id]
            graph2_features = embeddings[graph2_id]

            pair_features = np.concatenate([graph1_features, graph2_features])

            X.append(pair_features)
            y.append(label)

        return np.array(X), np.array(y)

    def create_model(self, input_shape):
        """
        Create a multi-layer perceptron model based on dynamic hyperparameters.
        The last layer is always an output layer with sigmoid activation for binary classification.

        Parameters:
        - input_shape: Shape of the input features

        Returns:
        - A compiled Keras Model
        """
        inputs = layers.Input(shape=input_shape)

        # Initial layer (first layer after input)
        x = inputs

        # Add hidden layers dynamically based on hyperparameters
        for i in range(self.hyperparameters['num_hidden_layers']):
            layer_config = self.hyperparameters['hidden_layers'][i]
            x = layers.Dense(layer_config['neurons'], activation='relu')(x)

            # Apply dropout if specified
            if layer_config['dropout'] > 0:
                x = layers.Dropout(layer_config['dropout'])(x)

        # Output layer (binary classification)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        # Create and compile the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def process_training(self):
        """
        Train the model with the specified hyperparameters.
        """
        X, y = self.prepare_dataset()

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        self.X_test = X_test
        self.y_test = y_test

        # Create the model
        self.model = self.create_model(input_shape=(X_train.shape[1],))

        # Set up callbacks
        callback_list = []

        # Add early stopping if enabled
        if self.hyperparameters['early_stopping']:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.hyperparameters['patience'],
                restore_best_weights=True
            )
            callback_list.append(early_stopping)

        # Model checkpoint to save the best model
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=os.path.join(self.saved_models_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        )
        callback_list.append(model_checkpoint)

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.hyperparameters['num_epochs'],
            batch_size=self.hyperparameters['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callback_list,
            verbose=1
        )

        self.set_history(history)
        self.set_y_test(y_test)

        # Evaluate on test set
        self.evaluate_model()

        return history

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Make predictions
        self.y_prob = self.model.predict(self.X_test)
        self.y_pred = (self.y_prob > 0.5).astype(int).flatten()

        self.set_y_pred(self.y_pred)
        self.set_y_prob(self.y_prob)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_prob)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

        return metrics

    def get_hyperparameters(self):
        return self.hyperparameters

    def set_X_test(self, X_test):
        self.X_test = X_test

    def get_X_test(self):
        return self.X_test

    def set_y_test(self, y_test):
        self.y_test = y_test

    def get_y_test(self):
        return self.y_test

    def get_history(self):
        return self.history

    def set_history(self, history):
        self.history = history

    def set_y_pred(self, y_pred):
        self.y_pred = y_pred

    def get_y_pred(self):
        return self.y_pred

    def set_y_prob(self, y_prob):
        self.y_prob = y_prob

    def get_y_prob(self):
        return self.y_prob

    def get_uuid(self):
        return self.uuid

    def save_model(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        model_filename = f"mlp_{self.uuid}.h5"
        model_path = os.path.join(self.saved_models_dir, model_filename)

        self.model.save(model_path)

        print(f"Model saved successfully to {model_path}")
        return model_path

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} does not exist")

        # Load the model
        self.model = tf.keras.models.load_model(model_path)

        print(f"Model loaded successfully from {model_path}")
        return self.model
