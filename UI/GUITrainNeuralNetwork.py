from tkinter import filedialog
import customtkinter as ctk

from UI.GUIUtil import GUIUtil


COLOR_GREY = "#7d8597"
COLOR_GREY_HOVER = "#979dac"
COLOR_GREEN = "#38b000"
COLOR_GREEN_HOVER = "#008000"
COLOR_RED = "#e63946"
COLOR_RED_HOVER = "#d90429"


class GUITrainNeuralNetwork:
    def __init__(self, root):
        self.root = root
        self.root.title("App")
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

    def __add_component(self, component_type, frame=None, grid_options=None, **kwargs):
        """
        Adds a component to the window.

        Args:
            component_type (str): The type of component to add (e.g., "Label", "Button", "Entry").
            grid_options (dict): A dictionary containing grid options (e.g., {"row": 0, "column": 0}).
            **kwargs: Additional keyword arguments to configure the component.
        """
        # Create the component based on the type
        if component_type == "Label":
            component = ctk.CTkLabel(frame, **kwargs)
        elif component_type == "Button":
            component = ctk.CTkButton(frame, **kwargs)
        elif component_type == "Entry":
            component = ctk.CTkEntry(frame, **kwargs)
        elif component_type == "Checkbutton":
            component = ctk.CTkCheckBox(frame, **kwargs)
        elif component_type == "Radiobutton":
            component = ctk.CTkRadioButton(frame, **kwargs)
        elif component_type == "Combobox":
            component = ctk.CTkComboBox(frame, **kwargs)
        elif component_type == "Text":
            component = ctk.CTkTextbox(frame, **kwargs)
        else:
            raise ValueError(f"Unsupported component type: {component_type}")

        # Place the component in the grid
        if grid_options:
            component.grid(**grid_options)
        else:
            component.pack()  # Default to pack if no grid options are provided

        return component

    def __goBackToOptions(self):
        self.guiUtil.removeWindow(root=self.root)
        # Create the new GUI in the same root window
        from UI.GUIChooseOptions import GUIChooseOptions
        app = GUIChooseOptions(self.root)
        app.run()

    def __addHeader(self):
        self.__add_component(
            component_type="Button",
            frame=self.root,
            text="< Back to options",
            command=lambda: self.__goBackToOptions(),
            grid_options={"row": 0, "column": 0, "columnspan": 1, "sticky": "n", "pady": 10},
            font=self.root.smallFont,
            fg_color=COLOR_GREY,
            hover_color=COLOR_GREY_HOVER,
            width=30,
            height=20


        )
        self.__add_component(
            component_type="Label",
            frame=self.root,
            text="Train Neural Network",
            grid_options={"row": 0, "column": 0, "columnspan": 7, "sticky": "n", "pady": 10},
            font=self.root.fontTitle
        )

    def __select_directory(self, entry, button=None):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, ctk.END)
            entry.insert(ctk.END, path)

            if button:
                button.configure(state="normal")

    def __removeDirectory(self, entry, button=None):
        entry.delete(0, ctk.END)
        if button:
            button.configure(state="disabled")

    def __createProcessingDataFrame(self):
        """Input"""

        self.__add_component(
            component_type="Label",
            frame=self.process_data_frame,
            text="Input directory",
            grid_options={"row": 1, "column": 0},
            font=self.root.font,
            anchor="center"
        )

        self.input_entry = self.__add_component(
            component_type="Entry",
            frame=self.process_data_frame,
            grid_options={"row": 2, "column": 0, "sticky": "w", "padx": 10},
            font=self.root.font,
            width=125,
            height=20
        )

        self.__add_component(
            component_type="Button",
            frame=self.process_data_frame,
            text="Select input directory",
            grid_options={"row": 3, "column": 0, "sticky": "w", "padx": 10, "pady": 5},
            font=self.root.font,
            width=50,
            command=lambda: self.__select_directory(self.input_entry, self.process_files_button)
        )

        self.__add_component(
            component_type="Button",
            frame=self.process_data_frame,
            text="Reset input directory",
            grid_options={"row": 4, "column": 0, "sticky": "w", "padx": 10},
            font=self.root.font,
            width=50,
            fg_color=COLOR_RED,
            hover_color=COLOR_RED_HOVER,
            command=lambda: self.__removeDirectory(self.input_entry, self.process_files_button)
        )

        """Output"""

        self.__add_component(
            component_type="Label",
            frame=self.process_data_frame,
            text="Output directory",
            grid_options={"row": 1, "column": 1, "sticky": "w", "padx": 20, "pady": 5},
            font=self.root.font,
            anchor="w"
        )

        self.output_entry = self.__add_component(
            component_type="Entry",
            frame=self.process_data_frame,
            grid_options={"row": 2, "column": 1, "sticky": "w", "padx": (0, 10), "pady": 5},
            font=self.root.font,
            width=130,
            height=20
        )

        self.__add_component(
            component_type="Button",
            frame=self.process_data_frame,
            text="Select output directory",
            grid_options={"row": 3, "column": 1, "sticky": "w", "pady": 5},
            font=self.root.font,
            width=50,
            command=lambda: self.__select_directory(self.output_entry)
        )

        self.__add_component(
            component_type="Button",
            frame=self.process_data_frame,
            text="Reset output directory",
            grid_options={"row": 4, "column": 1, "sticky": "w"},
            font=self.root.font,
            width=50,
            fg_color=COLOR_RED,
            hover_color=COLOR_RED_HOVER,
            command=lambda: self.__removeDirectory(self.output_entry)
        )

        """Is Orca files checkbox"""

        self.__add_component(
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="Data are Orca files",
            grid_options={"row": 5, "column": 0, "sticky": "w", "padx": 10, "pady": (15, 0)},
            font=self.root.font,
            checkbox_width=18,
            checkbox_height=17,
            corner_radius=7,
            fg_color=COLOR_GREEN,
            hover_color=COLOR_GREEN_HOVER,
            border_width=2
        )

        self.process_files_button = self.__add_component(
            component_type="Button",
            frame=self.process_data_frame,
            text="Process files",
            grid_options={"row": 6, "column": 0, "sticky": "ew", "padx": 10, "pady": (15, 0)},
            font=self.root.font,
            width=50,
            height=25,
            fg_color=COLOR_GREY,
            hover_color=COLOR_GREY_HOVER,
            state="disabled",
            command="",
        )

        self.__add_component(
            component_type="Button",
            text="Show graphs",
            frame=self.process_data_frame,
            grid_options={"row": 6, "column": 1, "sticky": "ew", "padx": (5,10), "pady": (15, 0)},
            font=self.root.font,
            width=50,
            height=25,
            fg_color=COLOR_GREY,
            hover_color=COLOR_GREY_HOVER,
            state="disabled",
            command="",
        )

    def run(self):
        self.__addHeader()
        self.__createProcessingDataFrame()


# if __name__ == "__main__":
#     root = ctk.CTk()  # Use CTk instead of Tk for the main window
#     app = GUITrainNeuralNetwork(root)
#     app.run()