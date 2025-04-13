import joblib
from tensorflow.keras.models import load_model

from BusinessLogic.Exception.CustomException import CustomException


class NeuralNetworkModelUtils:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def load_model(self, option_model):
        if option_model.endswith('.pkl'):
            model = joblib.load(self.model_dir + '/' + option_model, 'r+')
        elif option_model.endswith('.h5'):
            model = load_model(self.model_dir + '/' + option_model)
        else:
            raise CustomException("Unknown model type - " + option_model + ". Should be .pkl or .h5")

        print("model:", model)
        print("loading model from " + self.model_dir + '/' + option_model)

        return model

    def save_model(self, model, model_name):
        print("saving model to " + self.model_dir + '/' + model_name)
        joblib.dump(model, self.model_dir + '/' + model_name)
