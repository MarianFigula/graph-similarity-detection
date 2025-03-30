import uuid
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from BusinessLogic.DataNormaliser.DataNormaliser import DataNormaliser

# TODO metoda pre ukladanie modelu aj do vlastneho fodera s vizualizaciami
class RandomForestClassifierGraphSimilarity:
    def __init__(self, graphlet_counts, similarity_measures, hyperparameters):
        self.hyperparameters = hyperparameters
        self.graphlet_counts = graphlet_counts
        self.similarity_measures = similarity_measures
        self.saved_models_dir = 'MachineLearningData/saved_models'
        self.model = None

        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_prob = None

        self.uuid = uuid.uuid4().int

    def prepare_dateset(self):
        if 'Unnamed: 0' in self.graphlet_counts.columns:
            self.graphlet_counts = self.graphlet_counts.drop('Unnamed: 0', axis=1)

        graphlet_df = DataNormaliser(self.graphlet_counts).percentage_normalisation()

        if 'Unnamed: 0' in self.similarity_measures.columns:
            self.similarity_measures = self.similarity_measures.drop('Unnamed: 0', axis=1)

        graphlet_dict = {graphlet_df.columns[i]: graphlet_df.iloc[:, i].values for i in range(len(graphlet_df.columns))}

        X = []
        y = []

        for _, row in self.similarity_measures.iterrows():
            graph1_id = row['Graph1']
            graph2_id = row['Graph2']
            label = row['Label']

            graph1_features = graphlet_dict[graph1_id]
            graph2_features = graphlet_dict[graph2_id]

            pair_features = np.concatenate([graph1_features, graph2_features])

            X.append(pair_features)
            y.append(label)

        return np.array(X), np.array(y)

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.set_y_test(y_test)
        self.set_X_test(X_test)

        model = RandomForestClassifier(**self.hyperparameters, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        self.model = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        self.set_y_pred(y_pred)
        self.set_y_prob(y_prob)

        return X_test, y_test

    def set_X_test(self, X_test):
        self.X_test = X_test

    def get_X_test(self):
        return self.X_test

    def set_y_test(self, y_test):
        self.y_test = y_test

    def get_y_test(self):
        return self.y_test

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

        joblib.dump(self.model, f'{self.saved_models_dir}/rf_{self.uuid}.joblib')

        print(f"Model saved as {self.saved_models_dir}/rf_{self.uuid}.joblib")
        return True



    def process_training(self):
        X, y = self.prepare_dateset()
        self.train_model(X, y)
