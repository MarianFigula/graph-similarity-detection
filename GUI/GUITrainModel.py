import os
from tkinter import filedialog, IntVar
import customtkinter as ctk

from BusinessLogic.DataVisualiser.DataVisualiser import DataVisualiser
from BusinessLogic.ProcessFiles.SimilarityHandler import SimilarityHandler
from BusinessLogic.ProcessFiles.SnapShotOfGraphletsAsGraph import SnapShotOfGraphletsAsGraph
from GUI.Partial.GUIMlpClassifier import GUIMlpClassifier
from GUI.Partial.GUIRandomForestClassifier import GUIRandomForestClassifier
from GUI.GUIUtil import GUIUtil
import GUI.GUIConstants as guiconst


class GUITrainModel:
    def __init__(self, root):
        self.root = root
        self.root.title("Train model")
        self.root.fontTitle = ("Lato", 16)
        self.root.font = ("Lato", 12)
        self.root.fontMiddle = ("Lato", 13)
        self.root.smallFont = ("Lato", 10)
        self.guiUtil = GUIUtil()

        root.resizable(True, True)
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()

        w = 1200  # width for the Tk root
        h = 700  # height for the Tk root
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.train_model_width = w - 2 * 40
        self.train_model_frame = ctk.CTkFrame(self.root, width=400, height=650)
        self.train_model_frame.grid(row=2, column=0, padx=(40, 20), pady=10, sticky="nsew")

        self.train_model_frame.grid_columnconfigure(0, weight=1)
        self.train_model_frame.grid_columnconfigure(1, weight=1)
        self.train_model_frame.grid_propagate(False)

        # self.hidden_layers_frame = ctk.CTkFrame(self.root, width=150, height=350)
        # self.hidden_layers_frame.grid(row=2, column=1, padx=(0,40), pady=240, sticky="nsew")
        # self.guiUtil.add_component(
        #     self,
        #     component_type="Label",
        #     frame=self.hidden_layers_frame,
        #     text="Hidden layers",
        #     grid_options={"row": 0, "column": 0, "columnspan": 2, "sticky": "ew"},
        #     font=self.root.font
        # )
        # self.hidden_layers_frame.grid_propagate(False)
        # self.hidden_layers_frame.grid_columnconfigure(0, weight=1)
        # self.hidden_layers_frame.grid_columnconfigure(1, weight=1)

        self.mlp_hyperparameters = None

    def __goBackToOptions(self):
        self.guiUtil.removeWindow(root=self.root)
        from GUI.GUIChooseOptions import GUIChooseOptions
        app = GUIChooseOptions(self.root)
        app.run()

    def __handle_optionMenu_callback(self, choice):
        self.selected_model = choice
        self.__createHyperparametersBasedOnModel()


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
            grid_options={"row": 0, "column": 1, "sticky": "ne", "pady": 10, "padx": 20},
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
            text="Train Model",
            grid_options={"row": 1, "column": 0, "columnspan": 2, "sticky": "ew", "pady": 5},
            font=self.root.fontTitle
        )

    def __createInputData(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.train_model_frame,
            text="Graphlet Distribution (.csv)",
            grid_options={"row": 0, "column": 0, "sticky": "n", "pady": 5},
            font=self.root.font
        )

        self.graphlet_distribution_entry = self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.train_model_frame,
            grid_options={"row": 1, "column": 0, "sticky": "n"},
            font=self.root.font,
            width=200,
            height=20
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.train_model_frame,
            text=f"Select graphlet distribution",
            grid_options={"row": 2, "column": 0, "sticky": "n", "pady": 10},
            font=self.root.font,
            width=150,
            height=25,
            command=lambda: print("Button Clicked")
        )

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.train_model_frame,
            text="Similarities (.csv)",
            grid_options={"row": 0, "column": 1, "sticky": "n", "pady": 5},
            font=self.root.font
        )

        self.similarities_entry = self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.train_model_frame,
            grid_options={"row": 1, "column": 1, "sticky": "n"},
            font=self.root.font,
            width=200,
            height=20
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.train_model_frame,
            text=f"Select graph similarities",
            grid_options={"row": 2, "column": 1, "sticky": "n", "pady": 10},
            font=self.root.font,
            width=150,
            height=25,
            command=lambda: print("Button Clicked")
        )

        self.guiUtil.create_horizontal_line(self.train_model_frame, width=self.train_model_width - 20, row=3, column=0,
                                            padx=15, pady=5, columnspan=2, sticky="ew")


    def __createChooseModel(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.train_model_frame,
            text="Select model",
            grid_options={"row": 5, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.fontMiddle
        )

        self.modelOptionMenu = self.guiUtil.add_component(
            self,
            component_type="OptionMenu",
            frame=self.train_model_frame,
            grid_options={"row": 6, "column": 0, "columnspan": 2, "sticky": "n", "padx": 30, "pady": 5},
            font=self.root.font,
            width=200,
            height=25,
            values=["Random Forest Classifier", "MLP Classifier"],
            state="disabled",
            command=self.__handle_optionMenu_callback,
        )

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.train_model_frame,
            text="Hyperparameters",
            grid_options={"row": 7, "column": 0, "sticky": "w", "padx": 30,  "pady": 5},
            font=self.root.fontMiddle
        )

        # self.hyperparameters_frame = ctk.CTkFrame(self.train_model, width=self.train_model_width - 40, height=200)
        # self.hyperparameters_frame.grid(row=8, column=0, columnspan=2, padx=30, pady=10, sticky="nsew")

    def __createHyperparametersBasedOnModel(self):
        # Clear any existing hyperparameter widgets
        for widget in self.train_model_frame.winfo_children():
            if widget.grid_info().get('row', 0) >= 7:
                widget.destroy()

        # Clean up MLP hidden layers frame if it exists
        if hasattr(self, 'mlp_hyperparameters') and self.mlp_hyperparameters is not None:
            if hasattr(self.mlp_hyperparameters,
                       'hidden_layers_frame') and self.mlp_hyperparameters.hidden_layers_frame is not None:
                self.mlp_hyperparameters.hidden_layers_frame.destroy()
                self.mlp_hyperparameters.hidden_layers_frame = None

        if self.modelOptionMenu.get() == "Random Forest Classifier":
            self.hyperparameters_rf = GUIRandomForestClassifier(self.train_model_frame, self.guiUtil, self.root)
            self.mlp_hyperparameters = None

        elif self.modelOptionMenu.get() == "MLP Classifier":
            self.mlp_hyperparameters = GUIMlpClassifier(
                parent=self.train_model_frame,
                gui_util=self.guiUtil,
                root=self.root,
                max_hidden_layers=10
            )

    def run(self):
        self.__addHeader()
        self.__createInputData()
        self.__createChooseModel()
        self.__createHyperparametersBasedOnModel()
        # Enable the model selection dropdown
        self.modelOptionMenu.configure(state="normal")

        # Set a default model selection
        # self.modelOptionMenu.set("MLP Classifier")
        # Initialize the hyperparameters based on default selection
        self.__createHyperparametersBasedOnModel()

        self.root.mainloop()
        self.root.mainloop()

if __name__ == "__main__":
    root = ctk.CTk()
    app = GUITrainModel(root)
    app.run()