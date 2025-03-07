import os
import numpy as np
import networkx as nx
import pandas as pd
import time
import shutil
from itertools import combinations
from netrd.distance.netsimile import feature_extraction, graph_signature
from scipy.spatial.distance import canberra

FEATURES_CACHE_DIR = "../netSimFeaturesCache"
TARGET_DIRECTORY = "../netSimFiles"


class NetSimileModel:

    def __init__(self, path):
        """
        :param path: path to net simile input (.in) files
        """
        self.path = path
        self.result_df = pd.DataFrame()

    def remove_first_line(self):
        if os.path.exists(TARGET_DIRECTORY):
            shutil.rmtree(TARGET_DIRECTORY)
        os.makedirs(TARGET_DIRECTORY)

        for file in os.listdir(self.path):
            if file.endswith(".in"):
                input_path = os.path.join(self.path, file)
                output_path = os.path.join(TARGET_DIRECTORY, file)

                # Skip the first line and write the rest
                with open(input_path, "r") as infile, open(output_path, "w") as outfile:
                    next(infile)  # Skip the first line
                    for line in infile:
                        outfile.write(line)

    def get_graph_features(self, graph_file):
        """Get graph features, either from cache or by computing."""
        feature_file = os.path.join(FEATURES_CACHE_DIR, f"{graph_file}.npy")

        if os.path.exists(feature_file):
            # Load features from cache
            print(f"loading features {feature_file} of graph file {graph_file} from storage")
            return np.load(feature_file)

        # Compute features and save them
        graph_path = os.path.join(TARGET_DIRECTORY, graph_file)
        graph = nx.read_edgelist(graph_path)
        features = feature_extraction(graph)
        np.save(feature_file, features)
        return features

    def calculate_net_simile(self):
        print("starting calculations with timer")
        start_time = time.time()

        files = os.listdir(TARGET_DIRECTORY)
        results = []

        for file1, file2 in combinations(files, 2):
            print(f"processing {file1, file2}", end=" ")
            pair_start_time = time.time()

            # Get features for both graphs
            G1_features = self.get_graph_features(file1)
            G2_features = self.get_graph_features(file2)

            # Compute graph signatures
            G1_signature = graph_signature(G1_features)
            G2_signature = graph_signature(G2_features)


            # Calculate distance using Canberra metric
            # skusit hellingerovu vzdialenost
            distance = abs(canberra(G1_signature, G2_signature))

            # Extract graph names and store result
            graph_name_1 = os.path.splitext(file1)[0]
            graph_name_2 = os.path.splitext(file2)[0]

            results.append({
                'Graph1': graph_name_1,
                'Graph2': graph_name_2,
                'NetSimile': distance
            })

            pair_end_time = time.time()
            pair_elapsed_time = pair_end_time - pair_start_time
            print(f"Time taken for {file1} and {file2}: {pair_elapsed_time:.2f} seconds")

        self.result_df = pd.DataFrame(results)

        shutil.rmtree(TARGET_DIRECTORY)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Calculations completed in {elapsed_time:.2f} seconds")

    def compile_process(self):
        self.remove_first_line()
        self.calculate_net_simile()

        return self.result_df
