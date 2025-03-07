import os
import numpy as np
from keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class ResNetModel:
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.input_shape = (224, 224, 3)
        self.model = ResNet50(weights="imagenet", include_top=False, input_shape=self.input_shape)

    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=self.input_shape[:2])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = self.model.predict(img_array)
        return features.flatten()

    def getListOfImages(self):
        image_files = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return image_files

    def getImageFeatures(self, image_files):
        feature_dict = {}
        for img_name in image_files:
            img_path = os.path.join(self.img_dir, img_name)
            feature_dict[img_name] = self.extract_features(img_path)
        return feature_dict

    # TODO: pozriet ci rovnaky image s rovnakou cestou je v similarity csv
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    def computeSimilarity(self):
        similarity_list = []
        feature_dict = self.getImageFeatures(self.getListOfImages())

        for img1 in feature_dict.keys():
            img1_clean = os.path.splitext(img1)[0]  # Remove file extension

            for img2 in feature_dict.keys():
                img2_clean = os.path.splitext(img2)[0]  # Remove file extension

                if img1 < img2:
                    sim_score = cosine_similarity([feature_dict[img1]], [feature_dict[img2]])[0][0]

                    # Append result as a dictionary
                    similarity_list.append({
                        'Graph1': img1_clean,
                        'Graph2': img2_clean,
                        'ResNet': sim_score
                    })

        # Create DataFrame from list of dictionaries
        df = pd.DataFrame(similarity_list)
        return df
