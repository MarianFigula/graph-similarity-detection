import customtkinter as ctk


class GUIUtil:
    fontTitle = ("Lato", 16)

    def __init__(self):
        pass

    @staticmethod
    def add_component(self, component_type, frame=None, grid_options=None, **kwargs):
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
        elif component_type == "Progressbar":
            component = ctk.CTkProgressBar(frame, **kwargs)
        else:
            raise ValueError(f"Unsupported component type: {component_type}")

        # Place the component in the grid
        if grid_options:
            component.grid(**grid_options)
        else:
            component.pack()  # Default to pack if no grid options are provided

        return component

    @staticmethod
    def create_horizontal_line(frame=None, width=50, height=2, fg_color="gray", **kwargs):
        line = ctk.CTkFrame(frame,width=width, height=height, fg_color=fg_color)
        line.grid(**kwargs)  # No horizontal expansion
        return line

    @staticmethod
    def removeWindow(root):
        for widget in root.winfo_children():
            widget.destroy()
