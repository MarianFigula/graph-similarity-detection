from tkinter import IntVar

import customtkinter as ctk
import GUI.GUIConstants as guiconst


class GUIRandomForestClassifier:
    def __init__(self, parent, gui_util, root):
        """
        Creates a self-contained frame with Random Forest hyperparameter controls

        Args:
            parent: The parent widget
            gui_util: The utility class with the add_component method
            root: Font to use for labels
        """
        self.guiUtil = gui_util
        self.root = root
        self.controls = {}

        self.main_frame = ctk.CTkFrame(parent, width=600, height=400)
        self.main_frame.grid(row=7, column=0, columnspan=2, padx=(40, 20), pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_propagate(False)

        # Create and organize all hyperparameter controls
        self._create_hyperparameter_controls()

    def _create_hyperparameter_controls(self):
        # Number of trees
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Number of trees",
            grid_options={"row": 0, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.font
        )

        self.controls["number_of_trees"] = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self.main_frame,
            grid_options={"row": 0, "column": 0, "sticky": "e", "padx": 30, "pady": 5},
            min_value=50,
            max_value=500,
            default_value=100,
            step=30,
            data_type=int
        )

        # Max depth
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Max depth",
            grid_options={"row": 0, "column": 1, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.font
        )

        self.controls["max_depth"] = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self.main_frame,
            grid_options={"row": 0, "column": 1, "sticky": "e", "padx": 30, "pady": 5},
            min_value=1,
            max_value=50,
            default_value=10,
            step=1,
            data_type=int
        )

        # Min samples split
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Min samples split",
            grid_options={"row": 1, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.font
        )

        self.controls["min_samples_split"] = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self.main_frame,
            grid_options={"row": 1, "column": 0, "sticky": "e", "padx": 30, "pady": 5},
            min_value=2,
            max_value=20,
            default_value=2,
            step=1,
            data_type=int
        )

        # Min samples leaf
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Min samples leaf",
            grid_options={"row": 1, "column": 1, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.font
        )

        self.controls["min_samples_leaf"] = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self.main_frame,
            grid_options={"row": 1, "column": 1, "sticky": "e", "padx": 30, "pady": 5},
            min_value=1,
            max_value=15,
            default_value=1,
            step=1,
            data_type=int
        )

        # Batch size
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Batch size",
            grid_options={"row": 2, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.font
        )

        self.controls["batch_size"] = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self.main_frame,
            grid_options={"row": 2, "column": 0, "sticky": "e", "padx": 30, "pady": 5},
            min_value=8,
            max_value=256,
            default_value=32,
            step=8,
            data_type=int
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.main_frame,
            text="Train Model",
            grid_options={"row": 3, "column": 0, "columnspan": 2, "sticky": "n", "padx": 30, "pady": 5},
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            width=30,
            height=25,
            command=lambda: self.__trainModel()
        )

        self.guiUtil.create_horizontal_line(
            self.main_frame,
            width=380,
            row=4,
            column=0,
            padx=15,
            pady=5,
            columnspan=2,
            sticky="ew"
        )

        # Visualization section
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Visualization and Saving",
            grid_options={"row": 5, "column": 0, "sticky": "w", "padx": (30, 0), "pady": 5},
            font=self.root.fontMiddle
        )


        self.important_features_var = IntVar()
        self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.main_frame,
            text="Important Features",
            variable=self.important_features_var,
            grid_options={"row": 6, "column": 0, "sticky": "w", "padx": (30, 0), "pady": 5},
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            font=self.root.font,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2
        )

        self.confusion_matrix_var = IntVar()
        self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.main_frame,
            text="Confusion Matrix",
            variable=self.confusion_matrix_var,
            grid_options={"row": 6, "column": 0, "sticky": "e", "padx": (30, 0), "pady": 5},
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            font=self.root.font,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2
        )

        self.roc_curve_var = IntVar()
        self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.main_frame,
            text="ROC Curve",
            variable=self.roc_curve_var,
            grid_options={"row": 6, "column": 1, "sticky": "w", "padx": (30, 0), "pady": 5},
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            font=self.root.font,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2
        )

        self.classification_report_var = IntVar()
        self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.main_frame,
            text="Classification Report",
            variable=self.classification_report_var,
            grid_options={"row": 6, "column": 1, "sticky": "e", "padx": 30, "pady": 5},
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            font=self.root.font,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2
        )

        self.visualize_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.main_frame,
            text="Visualize",
            grid_options={"row": 7, "column": 0, "sticky": "n", "padx": 10, "pady": 20},
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            width=180,
            height=25,
            command=lambda: print("Visualizing...")
        )

        self.save_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.main_frame,
            text="Save Model",
            grid_options={"row": 7, "column": 1, "sticky": "n", "padx": 10, "pady": 20},
            width=180,
            height=25,
            command=lambda: print("Saving model...")
        )

    def __trainModel(self):
        print("Training model...")

    def get_hyperparameters(self):
        """
        Retrieves all hyperparameter values as a dictionary

        Returns:
            dict: Dictionary with hyperparameter names as keys and their values
        """
        return {
            "n_estimators": self.controls["number_of_trees"].get_value(),
            "max_depth": self.controls["max_depth"].get_value(),
            "min_samples_split": self.controls["min_samples_split"].get_value(),
            "min_samples_leaf": self.controls["min_samples_leaf"].get_value(),
            "batch_size": self.controls["batch_size"].get_value(),
        }

    def set_hyperparameters(self, params):
        """
        Sets hyperparameter values from a dictionary

        Args:
            params (dict): Dictionary with hyperparameter values to set
        """
        if "n_estimators" in params:
            self.controls["number_of_trees"].set_value(params["n_estimators"])
        if "max_depth" in params:
            self.controls["max_depth"].set_value(params["max_depth"])
        if "min_samples_split" in params:
            self.controls["min_samples_split"].set_value(params["min_samples_split"])
        if "min_samples_leaf" in params:
            self.controls["min_samples_leaf"].set_value(params["min_samples_leaf"])
        if "batch_size" in params:
            self.controls["batch_size"].set_value(params["batch_size"])
