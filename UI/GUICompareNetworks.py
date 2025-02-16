from tkinter import filedialog
import customtkinter as ctk

from UI.GUIUtil import GUIUtil
import GUIConstants as guiconst
class GUICompareNetworks:
    def __init__(self, root):
        self.root = root
        self.root.title("Compare Networks")
        self.root.geometry("1000x300")
        self.root.fontTitle = ("Lato", 16)
        self.root.font = ("Lato", 12)
        self.root.smallFont = ("Lato", 10)
        self.guiUtil = GUIUtil()
        self.root.grid_columnconfigure(1, weight=1)  # Make the column expand to center content
        self.root.grid_columnconfigure(2, weight=1)  # Make the column expand to center content
        self.root.grid_columnconfigure(3, weight=1)  # Make the column expand to center content
        self.root.grid_columnconfigure(4, weight=1)  # Make the column expand to center content

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        # Create a main frame to hold all components
        self.process_data_frame = ctk.CTkFrame(self.root, width=300, height=300)
        self.process_data_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ns")  # No horizontal expansion

    def __goBackToOptions(self):
        self.guiUtil.removeWindow(root=self.root)
        # Create the new GUI in the same root window
        from UI.GUIChooseOptions import GUIChooseOptions
        app = GUIChooseOptions(self.root)
        app.run()

    def __addHeader(self):
        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.root,
            text="< Back to options",
            command=lambda: self.__goBackToOptions(),
            grid_options={"row": 0, "column": 0, "columnspan": 1, "sticky": "n", "pady": 10},
            font=self.root.font,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            width=30,
            height=25

        )
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.root,
            text="Compare two networks",
            grid_options={"row": 0, "column": 0, "columnspan": 7, "sticky": "n", "pady": 5},
            font=self.root.fontTitle
        )

    def run(self):
        self.__addHeader()
        self.root.mainloop()

if __name__ == "__main__":
    root = ctk.CTk()  # Use CTk instead of Tk for the main window
    app = GUICompareNetworks(root)
    app.run()