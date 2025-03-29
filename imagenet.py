import os
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from matplotlib import pyplot as plt
from scipy.stats import shapiro, ks_1samp, kstest
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load Pretrained VGG16 Model (without fully connected layers)
vgg_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Path to directory containing scatter plot images
image_dir = "img_graphs"

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to VGG16 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize for VGG16

    features = model.predict(img_array)  # Extract features
    return features.flatten()  # Flatten feature map to 1D vector

# Get list of image files
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Extract features for all images
feature_dict = {}
for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    feature_dict[img_name] = extract_features(img_path, vgg_model)

# Compute similarity for all pairs
similarity_results = []
for i, img1 in enumerate(image_files):
    for j, img2 in enumerate(image_files):
        if i < j:  # Avoid duplicate comparisons
            sim_score = cosine_similarity([feature_dict[img1]], [feature_dict[img2]])[0][0]
            similarity_results.append((img1, img2, sim_score))

# Convert results to DataFrame
df = pd.DataFrame(similarity_results, columns=["Image1", "Image2", "Similarity"])
# df = df.sort_values(by="Similarity", ascending=False)  # Sort by highest similarity

# Save to CSV
df.to_csv("image_similarity_results_resnet_comparing_with_another.csv", index=False)

print("✅ Image similarity comparison completed! Results saved in 'image_similarity_results_normal.csv'.")

# Load similarity results from CSV
df = pd.read_csv("image_similarity_results_resnet.csv")

# Extract similarity scores
similarity_scores = df["Similarity"].values

# # Compute mean and standard deviation
# mean_sim = np.mean(similarity_scores)
# std_sim = np.std(similarity_scores)
# # Set threshold using Mean + 1.5 * Std (adjustable)
# threshold = mean_sim + 1.5 * std_sim
#
# print(f"📊 Computed Threshold: {threshold:.4f}")
#
# # Filter similar image pairs using threshold
# similar_pairs = df[df["Similarity"] >= threshold]
# print(f"✅ Found {len(similar_pairs)} similar image pairs using threshold {threshold:.4f}")
# # Save similar pairs to a new CSV
# similar_pairs.to_csv("similar_images.csv", index=False)

df = pd.read_csv("image_similarity_results_resnet.csv")


def kolmogorovTest(df):
    similarity_scores = df["Similarity"].values

    # Compute mean and standard deviation
    mean = np.mean(similarity_scores)
    std = np.std(similarity_scores)

    # Kolmogorov-Smirnov Test (Check against Normal Distribution)
    ks_stat, p_value = kstest(similarity_scores, 'norm', args=(mean, std))

    # Interpretation
    alpha = 0.05  # Significance level
    if p_value > alpha:
        print(f"Similarity scores follow a normal distribution (p-value = {p_value:.4f})")
    else:
        print(f"Similarity scores do NOT follow a normal distribution (p-value = {p_value:.4f})")
