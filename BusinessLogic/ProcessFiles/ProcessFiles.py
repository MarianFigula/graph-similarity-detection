import os
import pandas as pd
import subprocess

PATH = "C:/Users/majof/PycharmProjects/diplomovka/orca/orca.exe"
GRAPHLETS_COUNTS_FILE_NAME = "graphlet_counts.csv"
SIMILARITY_MEASURES_FILE_NAME = "similarity_measures.csv"


class ProcessFiles:
    def __init__(self, input_folder_path=None, output_folder_path=None, is_out_files=False):
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.is_out_files = is_out_files
        self.__orbit_counts_df = pd.DataFrame()
        self.__input_files = []

    def __read_folder(self):
        file_paths = []
        try:
            for root, _, files in os.walk(self.input_folder_path):
                for file in files:
                    file_path = os.path.join(root, file).replace("\\", "/")
                    # ak to je orca tak su to .out subory
                    if file_path.endswith(".out" if self.is_out_files else ".in"):
                        file_paths.append(file_path)

        except Exception as e:
            print(f"An error occurred: {e}")

        print("file_paths:", file_paths)
        return file_paths

    def __process_in_files_using_orca(self):
        count_orbits = pd.DataFrame()

        for file in self.__input_files:
            print(f"Processing {file}")
            file_name = file.split("/")[-1].replace(".in", "")

            print(self.output_folder_path)
            if self.output_folder_path is not None and self.output_folder_path != "":
                output_file = f"{self.output_folder_path}/{file_name}.out"
            else:
                output_file = f"{self.input_folder_path}/{file_name}.out"

            print(f"Running command: {PATH} node 5 {file} {output_file}")

            subprocess.call([PATH, "node", "5", file, output_file])
            print("\n")

            count_orbits = pd.concat(
                [count_orbits, self.__sum_orbits(file_name=output_file, column_name=file_name)], axis=1)

        return count_orbits

    def __process_out_files(self):
        count_orbits = pd.DataFrame()

        for file in self.__input_files:
            print(f"Processing {file}")
            file_name = file.split("/")[-1].replace(".out", "")

            count_orbits = pd.concat(
                [count_orbits, self.__sum_orbits(file_name=file, column_name=file_name)], axis=1)

        return count_orbits

    def __sum_orbits(self, file_name, column_name):
        print(file_name)
        df = pd.read_csv(file_name, sep=" ", header=None)
        column_sums = df.sum(axis=0)
        final_sums = pd.DataFrame({f"{column_name}": [0] * 30})

        final_sums.loc[0, column_name] = round(column_sums[0] / 2)
        final_sums.loc[1, column_name] = column_sums[2]
        final_sums.loc[2, column_name] = round(column_sums[3] / 3)
        final_sums.loc[3, column_name] = round(column_sums[4] / 2)
        final_sums.loc[4, column_name] = column_sums[7]
        final_sums.loc[5, column_name] = round(column_sums[8] / 4)
        final_sums.loc[6, column_name] = column_sums[9]
        final_sums.loc[7, column_name] = round(column_sums[12] / 2)
        final_sums.loc[8, column_name] = round(column_sums[14] / 4)
        final_sums.loc[9, column_name] = column_sums[17]
        final_sums.loc[10, column_name] = column_sums[18]
        final_sums.loc[11, column_name] = column_sums[23]
        final_sums.loc[12, column_name] = column_sums[25]
        final_sums.loc[13, column_name] = column_sums[27]
        final_sums.loc[14, column_name] = column_sums[33]
        final_sums.loc[15, column_name] = round(column_sums[34] / 5)
        final_sums.loc[16, column_name] = column_sums[35]
        final_sums.loc[17, column_name] = column_sums[39]
        final_sums.loc[18, column_name] = column_sums[44]
        final_sums.loc[19, column_name] = column_sums[45]
        final_sums.loc[20, column_name] = round(column_sums[50] / 2)
        final_sums.loc[21, column_name] = column_sums[52]
        final_sums.loc[22, column_name] = round(column_sums[55] / 2)
        final_sums.loc[23, column_name] = column_sums[56]
        final_sums.loc[24, column_name] = column_sums[61]
        final_sums.loc[25, column_name] = column_sums[62]
        final_sums.loc[26, column_name] = column_sums[65]
        final_sums.loc[27, column_name] = column_sums[69]
        final_sums.loc[28, column_name] = round(column_sums[70] / 2)
        final_sums.loc[29, column_name] = round(column_sums[72] / 5)

        return final_sums

    def process(self):
        # print(self.is_out_files)
        self.__input_files = self.__read_folder()
        if not self.is_out_files:
            self.__orbit_counts_df = self.__process_in_files_using_orca()
        else:
            self.__orbit_counts_df = self.__process_out_files()

        if self.output_folder_path is not None:
            self.__orbit_counts_df.to_csv(
                f"{self.output_folder_path}/{GRAPHLETS_COUNTS_FILE_NAME}",
                encoding="utf-8",
            )
