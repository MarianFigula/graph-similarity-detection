import customtkinter as ctk

from GUI.GUITrainNeuralNetwork import GUITrainNeuralNetwork
from GUI.GUICompareNetworks import GUICompareNetworks
from GUI.GUIUtil import GUIUtil


class GUIChooseOptions:
    def __init__(self, root):
        self.root = root
        self.root.title("Choose options")
        self.root.geometry("350x80")
        self.root.fontTitle = ("Lato", 16)
        self.guiUtil = GUIUtil()

        ws = root.winfo_screenwidth()  # width of the screen
        hs = root.winfo_screenheight()  # height of the screen

        w = 350  # width for the Tk root
        h = 80  # height for the Tk root
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)

        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

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
            grid_options={"row": 1, "column": 0,  "padx": 20},
            text="Compare 2 networks",
            command=lambda: self.__runCompareNetworks()
        )
        self.guiUtil.add_component(
            self.root,
            component_type="Button",
            grid_options={"row": 1, "column": 1, "sticky": "e"},
            text="Train Neural Network",
            command=lambda: self.__runTrainNeuralNetwork()
        )

    def __runCompareNetworks(self):
        self.guiUtil.removeWindow(root=self.root)
        app = GUICompareNetworks(self.root)
        app.run()
    def __runTrainNeuralNetwork(self):
        self.guiUtil.removeWindow(root=self.root)

        # Create the new GUI in the same root window
        app = GUITrainNeuralNetwork(self.root)
        app.run()

    def run(self):
        self.__addTitle()
        self.__createOptions()
        self.root.mainloop()

if __name__ == "__main__":
    # TODO: testnut ci po otvoreni a zvoleny moznost sa pekne zobrazi obrazovka
    root = ctk.CTk()  # Use CTk instead of Tk for the main window
    app = GUIChooseOptions(root)
    app.run()
