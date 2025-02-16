import customtkinter as ctk

from UI.GUITrainNeuralNetwork import GUITrainNeuralNetwork
from UI.GUIUtil import GUIUtil


class GUIChooseOptions:
    def __init__(self, root):
        self.root = root
        self.root.title("Choose options")
        self.root.geometry("315x80")
        self.root.fontTitle = ("Lato", 16)
        self.guiUtil = GUIUtil()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")


    def __addTitle(self):
        self.guiUtil.add_component(
            self.root,
            component_type="Label",
            grid_options={"row": 0, "column": 0, "columnspan": 7, "sticky": "n","padx": 30, "pady": 5},
            text="Choose an option you want to perform:",
            font=self.guiUtil.fontTitle,
            anchor="center"
        )

    def __createOptions(self):
        self.guiUtil.add_component(
            self.root,
            component_type="Button",
            grid_options={"row": 1, "column": 0, "sticky": "w", "padx": 10},
            text="Compare 2 networks",
            command=lambda: self.__runCompareNetworks()
        )
        self.guiUtil.add_component(
            self.root,
            component_type="Button",
            grid_options={"row": 1, "column": 1, "sticky": "w"},
            text="Train Neural Network",
            command=lambda: self.__runTrainNeuralNetwork()
        )

    def __runCompareNetworks(self):
        pass

    def __runTrainNeuralNetwork(self):
        self.guiUtil.removeWindow(root=self.root)

        # Create the new GUI in the same root window
        app = GUITrainNeuralNetwork(self.root)
        app.run()

    def run(self):
        self.__addTitle()
        self.__createOptions()
        self.root.mainloop()

