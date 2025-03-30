import os
import pickle
import joblib


class NeuralNetworkModelUtils:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def load_model(self, option_model):
        print("loading model from " + self.model_dir + '/' + option_model)
        model = joblib.load(self.model_dir + '/' + option_model, 'r+')

        return model

    def save_model(self, model, model_name):
        print("saving model to " + self.model_dir + '/' + model_name)
        joblib.dump(model, self.model_dir + '/' + model_name)
