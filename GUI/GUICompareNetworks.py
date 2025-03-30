import os
from tkinter import filedialog, IntVar
import customtkinter as ctk
import pandas as pd

from BusinessLogic.DataNormaliser.DataNormaliser import DataNormaliser
from BusinessLogic.Exception.CustomException import CustomException
from BusinessLogic.Exception.EmptyDataException import EmptyDataException
from GUI.GUIUtil import GUIUtil
import GUI.GUIConstants as guiconst
from BusinessLogic.GraphSimilarityML.NeuralNetworkPredictor import NeuralNetworkPredictor

class GUICompareNetworks:
    def __init__(self, root):
        self.root = root
        self.neuralNetworkPredictor = NeuralNetworkPredictor()

        self.root.title("Compare Networks")
        self.input_graphlet_df = None

        self.root.fontTitle = ("Lato", 16)
        self.root.font = ("Lato", 12)
        self.root.smallFont = ("Lato", 10)

        self.guiUtil = GUIUtil()
        self.root.grid_columnconfigure(1, weight=1)  # Make the column expand to center content
        self.root.grid_columnconfigure(2, weight=1)  # Make the column expand to center content
        self.root.grid_columnconfigure(3, weight=1)  # Make the column expand to center content
        self.root.grid_columnconfigure(4, weight=1)  # Make the column expand to center content

        root.resizable(False, False)
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()

        w = 400
        h = 600

        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)

        # TODO: odkomentovat
        # root.geometry('400x550')
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # Create a main frame to hold all components
        self.compare_networks_frame = ctk.CTkFrame(self.root, width=360, height=480)
        self.compare_networks_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ns")  # No horizontal expansion
        self.compare_networks_frame.grid_propagate(False)  # Prevent the frame from resizing based on its content

        self.compare_networks_frame.grid_columnconfigure(0, weight=1)  # Center content horizontally
        self.compare_networks_frame.grid_columnconfigure(1, weight=1)  # Center content horizontally

    def __goBackToOptions(self):
        self.guiUtil.removeWindow(root=self.root)
        # Create the new GUI in the same root window
        from GUI.GUIChooseOptions import GUIChooseOptions
        app = GUIChooseOptions(self.root)
        app.run()

    def __addHeader(self):
        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.root,
            text="< Back to options",
            command=lambda: self.__goBackToOptions(),
            grid_options={"row": 0, "column": 0, "sticky": "nw", "pady": 10, "padx": 15},
            font=self.root.font,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            width=30,
            height=25
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            text="?",
            grid_options={"row": 0, "column": 0, "sticky": "ne", "pady": 10, "padx": 20},
            font=self.root.font,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            width=30,
            height=25,
            command=lambda: self.guiUtil.openTopLevel("text"),
        )

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.root,
            text="Compare networks",
            grid_options={"row": 1, "column": 0, "columnspan": 1, "sticky": "n", "pady": 5},
            font=self.root.fontTitle
        )

    def __handleSelectDirectory(self, entry):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

        if not path:
            return

        entry.delete(0, ctk.END)
        entry.insert(ctk.END, path)

        self.guiUtil.setComponentNormalState(self.compare_button)
        self.guiUtil.setComponentNormalState(self.modelOptionMenu)

    def __handle_optionMenu_callback(self, choice):
        print(f"Selected model: {choice}")
        self.selected_model = choice

    def __handleComparison(self):
        error_msg = ""
        if self.selected_model is None:
            self.guiUtil.displayError(self.compare_networks_frame, "Please select a model")
            return
        print("comp, selected model: ", self.selected_model)

        if bool(self.compare_between_two_graphlets_val.get()):
            try:
                self.input_graphlet_df = pd.read_csv(self.input_entry.get())
                self.input_graphlet_df2 = pd.read_csv(self.input_entry_second.get())

                self.input_graphlet_df = DataNormaliser(self.input_graphlet_df).percentage_normalisation()
                self.input_graphlet_df2 = DataNormaliser(self.input_graphlet_df2).percentage_normalisation()

                self.result_df = self.neuralNetworkPredictor.predict_two_graphlet_distributions(
                    self.input_graphlet_df,
                    self.input_graphlet_df2,
                    self.selected_model)

                self.guiUtil.setComponentNormalState(self.download_button)
                self.guiUtil.setComponentNormalState(self.display_button)
            except EmptyDataException as e:
                print(e)
                error_msg = str(e)
            except CustomException as e:
                print(e)
                error_msg = str(e)
            except FileNotFoundError as e:
                print(e)
                error_msg = "File not found"
            except Exception as e:
                print(e)
                error_msg = "Error " + str(e)
            finally:
                if error_msg != "":
                    self.guiUtil.displayError(self.compare_networks_frame, error_msg, row=16, column=0, columnspan=2)

        # TODO: pre model treba model.save ale pre rfc staci cez joblib
        else:
            try:
                self.input_graphlet_df = pd.read_csv(self.input_entry.get())
                print("selected model: " + self.selected_model)
                self.input_graphlet_df = DataNormaliser(self.input_graphlet_df).percentage_normalisation()

                self.result_df = self.neuralNetworkPredictor.predict(self.input_graphlet_df, self.selected_model)

                self.guiUtil.setComponentNormalState(self.download_button)
                self.guiUtil.setComponentNormalState(self.display_button)
            except EmptyDataException as e:
                print(e)
                error_msg = str(e)
            except CustomException as e:
                print(e)
                error_msg = str(e)
            except FileNotFoundError as e:
                error_msg = "File not found"
                print(e)
            except Exception as e:
                print(e)
                error_msg = "Error " + str(e)
            finally:
                if error_msg != "":
                    self.guiUtil.displayError(self.compare_networks_frame, error_msg, row=16, column=0, columnspan=2)


    def toggle_second_graphlet_distribution(self):

        self.guiUtil.toggle_component(
            self.compare_between_two_graphlets_val,
            self.second_label,
            row=5,
            column=0,
            columnspan=2,
            sticky="n"
        )

        self.guiUtil.toggle_component(
            self.compare_between_two_graphlets_val,
            self.input_entry_second,
            row=6,
            column=0,
            columnspan=2,
            sticky="we",
            padx=30,
            pady=10
        )

        self.guiUtil.toggle_component(
            self.compare_between_two_graphlets_val,
            self.select_input_button_second,
            row=7,
            column=0,
            columnspan=2,
            sticky="we",
            padx=50,
            pady=10
        )

        if bool(self.compare_between_two_graphlets_val.get()):
            # Increase window and frame height
            new_root_height = 700
            new_frame_height = 570

            # Resize the root window
            ws = self.root.winfo_screenwidth()
            hs = self.root.winfo_screenheight()
            w = 400
            h = new_root_height
            x = (ws / 2) - (w / 2)
            y = (hs / 2) - (h / 2)
            self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))

            # Resize the frame
            self.compare_networks_frame.configure(height=new_frame_height)
        else:
            # Decrease window and frame height back to original
            original_root_height = 600
            original_frame_height = 450

            # Resize the root window
            ws = self.root.winfo_screenwidth()
            hs = self.root.winfo_screenheight()
            w = 400
            h = original_root_height
            x = (ws / 2) - (w / 2)
            y = (hs / 2) - (h / 2)
            self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))

            # Resize the frame
            self.compare_networks_frame.configure(height=original_frame_height)

    def __handleDownload(self):
        if self.result_df is None:
            return
        output_file = self.neuralNetworkPredictor.download_predictions(self.result_df, self.selected_model)

        self.download_complete_label.configure(text=f"Predictions saved in\n{output_file}!")
        self.compare_networks_frame.after(2000, lambda: self.guiUtil.reset_label(self.download_complete_label))

    def __getSavedModels(self):
        self.model_dir_path = "MachineLearningData/saved_models"

        if not os.path.exists(self.model_dir_path):
            os.makedirs(self.model_dir_path)

        files = [f for f in os.listdir(self.model_dir_path) if f.endswith(".pkl") or f.endswith(".h5")]

        self.selected_model = files[0]
        return files

    def __createGraphletDistributionInput(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.compare_networks_frame,
            text="Select graphlet distributions (.csv):",
            grid_options={"row": 2, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font
        )

        self.input_entry = self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.compare_networks_frame,
            grid_options={"row": 3, "column": 0, "columnspan": 2, "sticky": "we", "padx": 30, "pady": 10},
            font=self.root.font,
            height=20
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.compare_networks_frame,
            text="Select input file",
            grid_options={"row": 4, "column": 0, "columnspan": 2, "sticky": "we", "padx": 50, "pady": 5},
            font=self.root.font,
            width=30,
            height=25,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="normal",
            command=lambda: self.__handleSelectDirectory(self.input_entry)
        )

        self.second_label = self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.compare_networks_frame,
            text="Select second graphlet distributions (.csv):",
            grid_options={"row": 5, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font
        )

        self.input_entry_second = self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.compare_networks_frame,
            grid_options={"row": 6, "column": 0, "columnspan": 2, "sticky": "we", "padx": 30, "pady": 10},
            font=self.root.font,
            height=20,
            state="normal"
        )

        self.select_input_button_second = self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.compare_networks_frame,
            text="Select second input file",
            grid_options={"row": 7, "column": 0, "columnspan": 2, "sticky": "we", "padx": 50, "pady": (5, 10)},
            font=self.root.font,
            width=30,
            height=25,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="normal",
            command=lambda: self.__handleSelectDirectory(self.input_entry_second)
        )

        self.second_label.grid_remove()
        self.input_entry_second.grid_remove()
        self.select_input_button_second.grid_remove()

        self.compare_between_two_graphlets_val = IntVar()
        self.compare_between_two_graphlets = self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.compare_networks_frame,
            grid_options={"row": 8, "column": 0, "columnspan": 2, "sticky": "we", "padx": 30, "pady": 10},
            font=self.root.font,
            text="Compare between two graphlets",
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2,
            variable=self.compare_between_two_graphlets_val,
            command=lambda: self.toggle_second_graphlet_distribution()
        )

        self.guiUtil.create_horizontal_line(self.compare_networks_frame, width=300, column=0, row=9, columnspan=2, padx=5, pady=15, sticky="n")

    def __createChoosingModel(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.compare_networks_frame,
            text="Select model:",
            grid_options={"row": 10, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font
        )
        self.modelOptionMenu = self.guiUtil.add_component(
            self,
            component_type="OptionMenu",
            frame=self.compare_networks_frame,
            grid_options={"row": 11, "column": 0, "columnspan": 2, "sticky": "we", "padx": 30, "pady": 10},
            font=self.root.font,
            width=20,
            height=25,
            values=self.__getSavedModels(),
            state="disabled",
            command=self.__handle_optionMenu_callback,
        )
        self.compare_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.compare_networks_frame,
            text="Compare",
            grid_options={"row": 12, "column": 0, "columnspan": 2, "sticky": "we", "padx": 50, "pady": 10},
            font=self.root.font,
            width=30,
            height=30,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            state="disabled",
            command=lambda: self.__handleComparison()
        )
        self.guiUtil.create_horizontal_line(self.compare_networks_frame, width=300, column=0, row=13, columnspan=2, padx=5, pady=15, sticky="n")

    def __displayResults(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.compare_networks_frame,
            text="Results",
            grid_options={"row": 14, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font
        )
        self.download_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.compare_networks_frame,
            text="Download results",
            grid_options={"row": 15, "column": 0, "sticky": "we", "padx": 10, "pady": 10},
            font=self.root.font,
            width=30,
            height=30,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="disabled",
            command=lambda: self.__handleDownload()
        )
        self.display_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.compare_networks_frame,
            text="Display results",
            grid_options={"row": 15, "column": 1, "sticky": "we", "padx": 10, "pady": 10},
            font=self.root.font,
            width=30,
            height=30,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="disabled",
            command=lambda: self.neuralNetworkPredictor.display_predictions(self.result_df)
        )
        self.download_complete_label = self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.compare_networks_frame,
            text="",
            grid_options={"row": 16, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font,
            text_color=guiconst.COLOR_GREEN,
            wraplength=300
        )

    def run(self):
        self.__addHeader()
        self.__createGraphletDistributionInput()
        self.__createChoosingModel()
        self.__displayResults()
        self.root.mainloop()


if __name__ == "__main__":
    root = ctk.CTk()
    app = GUICompareNetworks(root)
    app.run()
