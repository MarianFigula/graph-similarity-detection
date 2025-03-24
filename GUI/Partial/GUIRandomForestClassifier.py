
import customtkinter as ctk
import GUI.GUIConstants as guiconst


class GUIRandomForestClassifier(ctk.CTkFrame):
    def __init__(self, master, guiUtil, font, **kwargs):
        """
        Creates a self-contained frame with Random Forest hyperparameter controls

        Args:
            master: The parent widget
            guiUtil: The utility class with the add_component method
            font: Font to use for labels
            **kwargs: Additional keyword arguments for the CTkFrame
        """
        super().__init__(master, **kwargs)

        self.guiUtil = guiUtil
        self.font = font
        self.controls = {}

        # Create and organize all hyperparameter controls
        self._create_hyperparameter_controls()

    def _create_hyperparameter_controls(self):
        # Number of trees
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self,
            text="Number of trees",
            grid_options={"row": 0, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            font=self.font
        )

        self.controls["number_of_trees"] = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self,
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
            frame=self,
            text="Max depth",
            grid_options={"row": 0, "column": 1, "sticky": "w", "padx": 30, "pady": 5},
            font=self.font
        )

        self.controls["max_depth"] = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self,
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
            frame=self,
            text="Min samples split",
            grid_options={"row": 1, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            font=self.font
        )

        self.controls["min_samples_split"] = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self,
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
            frame=self,
            text="Min samples leaf",
            grid_options={"row": 1, "column": 1, "sticky": "w", "padx": 30, "pady": 5},
            font=self.font
        )

        self.controls["min_samples_leaf"] = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self,
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
            frame=self,
            text="Batch size",
            grid_options={"row": 2, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            font=self.font
        )

        self.controls["batch_size"] = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self,
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
            frame=self,
            text="Train Model",
            grid_options={"row": 3, "column": 0, "columnspan": 2, "sticky": "n", "padx": 30, "pady": 5},
            font=self.font,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            width=30,
            height=25,
        )

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
