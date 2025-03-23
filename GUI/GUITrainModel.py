import os
from tkinter import filedialog, IntVar
import customtkinter as ctk

from BusinessLogic.DataVisualiser.DataVisualiser import DataVisualiser
from BusinessLogic.ProcessFiles.SimilarityHandler import SimilarityHandler
from BusinessLogic.ProcessFiles.SnapShotOfGraphletsAsGraph import SnapShotOfGraphletsAsGraph
from GUI.GUIUtil import GUIUtil
from BusinessLogic.ProcessFiles.ProcessInAndOutFiles import ProcessInAndOutFiles
import GUI.GUIConstants as guiconst


class GUITrainModel:
    def __init__(self, root):
        self.root = root
        self.root.title("Train model")
        self.root.fontTitle = ("Lato", 16)
        self.root.font = ("Lato", 12)
        self.root.smallFont = ("Lato", 10)
        self.guiUtil = GUIUtil()

        root.resizable(True, True)
        ws = root.winfo_screenwidth()  # width of the screen
        hs = root.winfo_screenheight()  # height of the screen

        w = 750  # width for the Tk root
        h = 500  # height for the Tk root
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)

        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root.grid_columnconfigure(1, weight=1)  # Make the column expand to center content
        self.root.grid_columnconfigure(2, weight=1)  # Make the column expand to center content

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.train_neural_network_frame = ctk.CTkFrame(self.root, width=w-2*40, height=300)
        self.train_neural_network_frame.grid(row=2, column=0, padx=40, pady=20, sticky="ns")  # No horizontal expansion

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
            text="Train Model",
            grid_options={"row": 1, "column": 0, "columnspan": 7, "sticky": "ew", "pady": 5},
            font=self.root.fontTitle
        )

    def run(self):
        self.__addHeader()
        self.root.mainloop()

if __name__ == "__main__":
    root = ctk.CTk()  # Use CTk instead of Tk for the main window
    app = GUITrainModel(root)
    app.run()