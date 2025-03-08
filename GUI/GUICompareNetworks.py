import os
from tkinter import filedialog
import customtkinter as ctk
import pandas as pd
from GUI.GUIUtil import GUIUtil
import GUI.GUIConstants as guiconst
from training_neural_network.NeuralNetworkPredictor import NeuralNetworkPredictor

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
        h = 550

        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)

        # TODO: odkomentovat
        root.geometry('400x550')
        # root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # Create a main frame to hold all components
        self.process_data_frame = ctk.CTkFrame(self.root, width=320, height=400)
        self.process_data_frame.grid(row=2, column=0, padx=40, pady=20, sticky="ns")  # No horizontal expansion
        self.process_data_frame.grid_propagate(False)  # Prevent the frame from resizing based on its content

        self.process_data_frame.grid_columnconfigure(0, weight=1)  # Center content horizontally
        self.process_data_frame.grid_columnconfigure(1, weight=1)  # Center content horizontally

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
        if path:
            entry.delete(0, ctk.END)
            entry.insert(ctk.END, path)

            self.guiUtil.setComponentNormalState(self.compare_button)
            self.guiUtil.setComponentNormalState(self.modelOptionMenu)

    def __handle_optionMenu_callback(self, choice):
        self.selected_model = choice
        print("optionmenu dropdown clicked:", self.selected_model)

    def __handleComparison(self):
        self.input_graphlet_df = pd.read_csv(self.input_entry.get())

        if self.selected_model is None:
            return

        self.result_df = self.neuralNetworkPredictor.predict(self.input_graphlet_df, self.selected_model)

        self.guiUtil.setComponentNormalState(self.download_button)
        self.guiUtil.setComponentNormalState(self.display_button)


    def __handleDownload(self):
        if self.result_df is None:
            return
        self.neuralNetworkPredictor.download_predictions(self.result_df)

        self.download_complete_label.configure(text="Download complete!")
        self.process_data_frame.after(2000, lambda: self.guiUtil.resetDownloadLabel(self.download_complete_label))

    def __getSavedModels(self):
        self.model_dir_path = "../training_neural_network/saved_models"

        if not os.path.exists(self.model_dir_path):
            os.makedirs(self.model_dir_path)

        files = [f for f in os.listdir(self.model_dir_path) if f.endswith(".pkl") and f.find("scaler") == -1]

        self.selected_model = files[0]
        return files

    def __createGraphletDistributionInput(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.process_data_frame,
            text="Select graphlet distributions (.csv):",
            grid_options={"row": 2, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font
        )

        self.input_entry = self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.process_data_frame,
            grid_options={"row": 3, "column": 0, "columnspan": 2, "sticky": "we", "padx": 30, "pady": 10},
            font=self.root.font,
            height=20
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.process_data_frame,
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

        self.guiUtil.create_horizontal_line(self.process_data_frame, width=300, column=0, row=5, columnspan=2, padx=5,
                                            pady=15, sticky="n")

    def __createChoosingModel(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.process_data_frame,
            text="Select model:",
            grid_options={"row": 6, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font
        )

        self.modelOptionMenu = self.guiUtil.add_component(
            self,
            component_type="OptionMenu",
            frame=self.process_data_frame,
            grid_options={"row": 7, "column": 0, "columnspan": 2, "sticky": "we", "padx": 30, "pady": 10},
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
            frame=self.process_data_frame,
            text="Compare",
            grid_options={"row": 8, "column": 0, "columnspan": 2, "sticky": "we", "padx": 50, "pady": 10},
            font=self.root.font,
            width=30,
            height=30,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            state="disabled",
            command=lambda: self.__handleComparison()
        )

        self.guiUtil.create_horizontal_line(self.process_data_frame, width=300, column=0, row=9, columnspan=2, padx=5,
                                            pady=15, sticky="n")

    def __displayResults(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.process_data_frame,
            text="Results",
            grid_options={"row": 10, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font
        )

        self.download_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Download results",
            grid_options={"row": 11, "column": 0, "sticky": "we", "padx": 10, "pady": 10},
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
            frame=self.process_data_frame,
            text="Display results",
            grid_options={"row": 11, "column": 1, "sticky": "we", "padx": 10, "pady": 10},
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
            frame=self.process_data_frame,
            text="",
            grid_options={"row": 12, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font,
            text_color=guiconst.COLOR_GREEN,
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
