import os
from tkinter import filedialog, IntVar
import customtkinter as ctk

from BusinessLogic.DataVisualiser.DataVisualiser import DataVisualiser
from BusinessLogic.ProcessFiles.SimilarityHandler import SimilarityHandler
from BusinessLogic.ProcessFiles.SnapShotOfGraphletsAsGraph import SnapShotOfGraphletsAsGraph
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

        w = 600  # width for the Tk root
        h = 500  # height for the Tk root
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.train_model_width = w - 2 * 40
        self.train_model = ctk.CTkFrame(self.root, width=self.train_model_width, height=300)
        self.train_model.grid(row=2, column=0, padx=40, pady=20, sticky="nsew")

        self.train_model.grid_columnconfigure(0, weight=1)
        self.train_model.grid_columnconfigure(1, weight=1)
        self.train_model.grid_propagate(False)

    def __goBackToOptions(self):
        self.guiUtil.removeWindow(root=self.root)
        from GUI.GUIChooseOptions import GUIChooseOptions
        app = GUIChooseOptions(self.root)
        app.run()

    def __handle_optionMenu_callback(self, choice):
        self.selected_model = choice


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
            text="Train Model",
            grid_options={"row": 1, "column": 0, "columnspan": 2, "sticky": "ew", "pady": 5},
            font=self.root.fontTitle
        )

    def __createInputData(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.train_model,
            text="Graphlet Distribution (.csv)",
            grid_options={"row": 0, "column": 0, "sticky": "n", "pady": 5},
            font=self.root.font
        )

        self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.train_model,
            grid_options={"row": 1, "column": 0, "sticky": "n"},
            font=self.root.font,
            width=150,
            height=20
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.train_model,
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
            frame=self.train_model,
            text="Similarities (.csv)",
            grid_options={"row": 0, "column": 1, "sticky": "n", "pady": 5},
            font=self.root.font
        )

        self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.train_model,
            grid_options={"row": 1, "column": 1, "sticky": "n"},
            font=self.root.font,
            height=20
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.train_model,
            text=f"Select graph similarities",
            grid_options={"row": 2, "column": 1, "sticky": "n", "pady": 10},
            font=self.root.font,
            width=150,
            height=25,
            command=lambda: print("Button Clicked")
        )

        self.guiUtil.create_horizontal_line(self.train_model, width=self.train_model_width - 20, row=3, column=0,
                                            padx=15, pady=(15, 10), columnspan=2, sticky="ew")


    def __createChooseModel(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.train_model,
            text="Select model",
            grid_options={"row": 5, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.fontMiddle
        )

        self.modelOptionMenu = self.guiUtil.add_component(
            self,
            component_type="OptionMenu",
            frame=self.train_model,
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
            frame=self.train_model,
            text="Hyperparameters",
            grid_options={"row": 7, "column": 0, "sticky": "w", "padx": 30,  "pady": 10},
            font=self.root.fontMiddle
        )


    def run(self):
        self.__addHeader()
        self.__createInputData()
        self.__createChooseModel()
        self.root.mainloop()

if __name__ == "__main__":
    root = ctk.CTk()
    app = GUITrainModel(root)
    app.run()