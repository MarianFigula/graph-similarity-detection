from itertools import combinations
import numpy as np
import pandas as pd
from keras.layers import Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC


class NetworkClass:
    def __init__(self, graphlet_counts_df):
        self.model = None
        self.input_dim = 30
        self.graphlet_counts_df = graphlet_counts_df
        self.train_graphlet_counts_df = pd.read_csv("graphlet_counts.csv")
        self.train_similarity_measures_df = pd.read_csv("filtered_measures.csv")

        # Apply StandardScaler to the graphlet counts dataframe and keep the DataFrame structure
        self.scaler = StandardScaler()
        self.train_graphlet_counts_df_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.train_graphlet_counts_df),
            columns=self.train_graphlet_counts_df.columns
        )

    def __generate_pairs_labels(self, df):
        self.train_similarity_measures_df = self.train_similarity_measures_df.sample(frac=1, random_state=42).reset_index(drop=True)
        pairs, labels = [], self.train_similarity_measures_df["Label by names"]

        for index, row in self.train_similarity_measures_df.iterrows():
            graph1 = row["Graph1"]
            graph2 = row["Graph2"]
            pairs.append([df[graph1].values, df[graph2].values])

        return np.array(pairs), np.array(labels)

    def __siamese_network(self):
        model = Sequential(
            [
                Dense(30, activation="relu", input_shape=(self.input_dim,)),
                Dropout(0.5),
                Dense(10, activation="relu"),
            ]
        )
        return model

    def __create_siamese_model(self):
        input_a = Input(shape=(self.input_dim,))
        input_b = Input(shape=(self.input_dim,))

        siamese_model = self.__siamese_network()
        encoded_a = siamese_model(input_a)
        encoded_b = siamese_model(input_b)

        similarity_score = Dense(1, activation="sigmoid")(
            Lambda(lambda x: K.abs(x[0] - x[1]))([encoded_a, encoded_b])
        )

        model = Model(inputs=[input_a, input_b], outputs=similarity_score)

        return model

    def __compile_model(self):
        self.model.compile(
            optimizer=Adam(),
            loss=BinaryCrossentropy(),
            metrics=["accuracy", AUC()]
        )

    def train_model(self, epochs=10, batch_size=30, validation_split=0.2):
        pairs, labels = self.__generate_pairs_labels(self.train_graphlet_counts_df_scaled)
        early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

        history = self.model.fit(
            [pairs[:, 0], pairs[:, 1]],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
        )
        return history

    def __generate_pairs(self):
        column_combinations = sorted(
            list(combinations(sorted(self.graphlet_counts_df.columns), 2))
        )
        pairs = []
        for col1, col2 in column_combinations:
            pairs.append([self.graphlet_counts_df[col1], self.graphlet_counts_df[col2]])
        return np.array(pairs)

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()

        plt.show()

    def plot_heatmap(self, similarity_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, cmap="coolwarm", annot=True)
        plt.title("Graph Similarity Heatmap")
        plt.show()

    def predict_similarity(self):
        self.model = self.__create_siamese_model()
        self.__compile_model()
        history = self.train_model()

        self.plot_training_history(history)

        # similarity_matrix = np.random.rand(10, 10)  # Replace this with actual similarity data
        # self.plot_heatmap(similarity_matrix)


# Assuming graphlet_counts_df is already loaded
NetworkClass(graphlet_counts_df=None).predict_similarity()
