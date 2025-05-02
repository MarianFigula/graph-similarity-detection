import customtkinter as ctk

from GUI.GUIProcessData import GUIProcessData
from GUI.GUICompareNetworks import GUICompareNetworks
from GUI.GUITrainModel import GUITrainModel
from GUI.GUIUtil import GUIUtil


class GUIChooseOptions:
    def __init__(self, root):
        self.root = root
        self.root.title("Choose option")
        self.root.font_title = ("Lato", 16)
        self.gui_util = GUIUtil()

        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()

        w = 350
        h = 215
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)

        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

    def __add_title(self):
        self.gui_util.add_component(
            self.root,
            component_type="Label",
            grid_options={"row": 0, "column": 0, "sticky": "n", "padx": 30, "pady": (10, 20)},
            text="Choose an option you want to perform",
            font=self.gui_util.font_title,
            anchor="center"
        )

    def __create_options(self):
        self.gui_util.add_component(
            self.root,
            component_type="Button",
            grid_options={"row": 1, "column": 0, "padx": 20, "pady": (0, 15)},
            text="Process data",
            command=lambda: self.__run_process_data()
        )

        self.gui_util.add_component(
            self.root,
            component_type="Button",
            grid_options={"row": 2, "column": 0, "padx": 20, "pady": (0, 15)},
            text="Train Model",
            command=lambda: self.__run_train_model()
        )

        self.gui_util.add_component(
            self.root,
            component_type="Button",
            grid_options={"row": 3, "column": 0, "padx": 20, "pady": (0, 15)},
            text="Compare 2 networks",
            command=lambda: self.__run_compare_networks()
        )

    def __run_compare_networks(self):
        self.gui_util.remove_window(root=self.root)
        app = GUICompareNetworks(self.root)
        app.run()

    def __run_train_model(self):
        self.gui_util.remove_window(root=self.root)
        app = GUITrainModel(self.root)
        app.run()

    def __run_process_data(self):
        self.gui_util.remove_window(root=self.root)
        app = GUIProcessData(self.root)
        app.run()

    def run(self):
        self.__add_title()
        self.__create_options()
        self.root.mainloop()

