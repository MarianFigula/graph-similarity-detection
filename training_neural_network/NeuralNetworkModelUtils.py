import os
import pickle


class NeuralNetworkModelUtils:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    # TODO: ukladat potom scaler tak aby to bolo - model_name_scaler
    def get_scaler(self, option_model):
        return option_model.split('.')[0] + '_scaler.pkl'
    def load_model(self, option_model):
        with open(self.model_dir + '/' + option_model, 'rb') as f:
            mlp = pickle.load(f)

        with open(self.model_dir + '/scalers/' + self.get_scaler(option_model), 'rb') as f:
            scaler = pickle.load(f)

        return mlp, scaler

    def save_model(self, mlp, scaler):
        # TODO: Trochu prerobit, popremyslat nad tym
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Save the model
        with open(self.model_dir + '/graph_similarity.pkl', 'wb') as f:
            pickle.dump(mlp, f)

        # Save the scaler
        with open(self.model_dir + '/graph_similarity_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Model and scaler saved to {self.model_dir}")

