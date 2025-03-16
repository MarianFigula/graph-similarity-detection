import customtkinter as ctk

from GUI.NumberInput import NumberInput


class GUIUtil:
    fontTitle = ("Lato", 16)

    def __init__(self):
        self.info_window = None  # Class-level variable to store the reference to the open window


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
        elif component_type == "OptionMenu":
            component = ctk.CTkOptionMenu(frame, **kwargs)
        elif component_type == "NumberInput":
            component = NumberInput(frame, **kwargs)
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

    @staticmethod
    def setComponentNormalState(component=None):
        if component is None:
            return

        component.configure(state="normal")

    @staticmethod
    def resetDownloadLabel(label_component):
        label_component.configure(text="")

    @staticmethod
    def toggle_component(checkbox_val, component, **kwargs):
        if bool(checkbox_val.get()):
            component.grid(**kwargs)
        else:
            component.grid_remove()

    def __create_info_top_level(self, text, **kwargs):
        # TODO: nastavit pozicia kde sa to bude zobrazovat
        info_window = ctk.CTkToplevel()
        info_window.title("Info")
        info_window.geometry("300x100")

        info_label = ctk.CTkLabel(info_window, text=text, **kwargs)
        info_label.grid(row=0, column=0, padx=10, pady=10)

        info_window.lift()
        info_window.focus_force()

        return info_window

    def openTopLevel(self, text=''):
        if self.info_window is None or not self.info_window.winfo_exists():
            self.info_window = self.__create_info_top_level(text)
        else:
            self.info_window.lift()
            self.info_window.focus_force()
