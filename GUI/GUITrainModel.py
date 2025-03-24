from tkinter import filedialog

import customtkinter as ctk
from GUI.Partial.GUIMlpClassifier import GUIMlpClassifier
from GUI.Partial.GUIRandomForestClassifier import GUIRandomForestClassifier
from GUI.GUIUtil import GUIUtil
import GUI.GUIConstants as guiconst


class GUITrainModel:
    def __init__(self, root):
        self.root = root
        self.root.title("Train model")
        self.root.fontTitle = ("Lato", 16)
        self.root.font = ("Lato", 12)
        self.root.fontMiddle = ("Lato", 13)
        self.root.smallFont = ("Lato", 10)
        self.guiUtil = GUIUtil()

        root.resizable(True, True)
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()

        w = 1200
        h = 750
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.train_model_width = w - 2 * 40
        self.train_model_frame = ctk.CTkFrame(self.root, width=self.train_model_width, height=600)
        self.train_model_frame.grid(row=2, column=0, padx=40, pady=10, sticky="nsew")

        self.train_model_frame.grid_columnconfigure(0, weight=1)
        self.train_model_frame.grid_columnconfigure(1, weight=1)
        self.train_model_frame.grid_propagate(False)

        self.should_enable_random_forest = False

        self.mlp_hyperparameters = None

    def resize_window(self, width, height, train_model_width=400):
        """
        Resize the root window while maintaining center position
        """
        # Get screen dimensions
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws / 2) - (width / 2)
        y = (hs / 2) - (height / 2)

        self.root.geometry('%dx%d+%d+%d' % (width, height, x, y))

        self.train_model_frame.configure(width=train_model_width, height=600)

        self.root.update_idletasks()

    def __goBackToOptions(self):
        self.guiUtil.removeWindow(root=self.root)
        from GUI.GUIChooseOptions import GUIChooseOptions
        app = GUIChooseOptions(self.root)
        app.run()

    def __handle_optionMenu_callback(self, choice):
        self.selected_model = choice
        self.__createHyperparametersBasedOnModel()

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

        self.info_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            text="?",
            grid_options={"row": 0, "column": 1, "sticky": "ne", "pady": 10, "padx": 20},
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
            grid_options={"row": 1, "column": 0, "columnspan": 2, "sticky": "ew", "pady": 5},
            font=self.root.fontTitle
        )

    def __handleSelectDirectory(self, main_entry: ctk.CTkEntry, second_entry: ctk.CTkEntry = None, component=None):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

        if not path:
            return

        main_entry.delete(0, ctk.END)
        main_entry.insert(ctk.END, path)

        if second_entry is None or second_entry.get() == "" or component is None:
            return

        component.configure(state="normal")

        if self.hyperparameters_rf is not None:
            self.hyperparameters_rf.enable_all_components()
            self.should_enable_random_forest = True

    def __createInputData(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.train_model_frame,
            text="Graphlet Distribution (.csv)",
            grid_options={"row": 0, "column": 0, "sticky": "n", "pady": 5},
            font=self.root.font
        )

        self.graphlet_distribution_entry = self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.train_model_frame,
            grid_options={"row": 1, "column": 0, "sticky": "n"},
            font=self.root.font,
            width=200,
            height=20,
            state="readonly",
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.train_model_frame,
            text=f"Select graphlet distribution",
            grid_options={"row": 2, "column": 0, "sticky": "n", "pady": 10},
            font=self.root.font,
            width=150,
            height=25,
            command=lambda: self.__handleSelectDirectory(self.graphlet_distribution_entry, self.similarities_entry,
                                                         self.modelOptionMenu)
        )

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.train_model_frame,
            text="Similarities (.csv)",
            grid_options={"row": 0, "column": 1, "sticky": "n", "pady": 5},
            font=self.root.font
        )

        self.similarities_entry = self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.train_model_frame,
            grid_options={"row": 1, "column": 1, "sticky": "n"},
            font=self.root.font,
            width=200,
            height=20,
            state="readonly",
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.train_model_frame,
            text=f"Select graph similarities",
            grid_options={"row": 2, "column": 1, "sticky": "n", "pady": 10},
            font=self.root.font,
            width=150,
            height=25,
            command=lambda: self.__handleSelectDirectory(self.similarities_entry, self.graphlet_distribution_entry,
                                                         self.modelOptionMenu)
        )

        self.guiUtil.create_horizontal_line(self.train_model_frame, width=self.train_model_width - 20, row=3, column=0,
                                            padx=15, pady=5, columnspan=2, sticky="ew")

    def __createChooseModel(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.train_model_frame,
            text="Select model",
            grid_options={"row": 5, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.fontMiddle
        )

        self.modelOptionMenu = self.guiUtil.add_component(
            self,
            component_type="OptionMenu",
            frame=self.train_model_frame,
            grid_options={"row": 6, "column": 0, "columnspan": 2, "sticky": "n", "padx": 30, "pady": 5},
            font=self.root.font,
            width=200,
            height=25,
            values=["Random Forest Classifier", "MLP Classifier"],
            state="disabled",
            command=self.__handle_optionMenu_callback,
        )

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.train_model_frame,
            text="Hyperparameters",
            grid_options={"row": 7, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.fontMiddle
        )

    def __setRandomForestGui(self):
        self.resize_window(800, 700, self.train_model_width)
        self.hyperparameters_rf = GUIRandomForestClassifier(self.train_model_frame, self.guiUtil, self.root)
        self.info_button.grid(row=0, column=0)

        if self.should_enable_random_forest:
            self.hyperparameters_rf.enable_all_components()

        self.mlp_hyperparameters = None

    def __setMlpGui(self):
        self.resize_window(1200, 700)
        self.info_button.grid(row=0, column=1)
        self.mlp_hyperparameters = GUIMlpClassifier(
            parent=self.train_model_frame,
            gui_util=self.guiUtil,
            root=self.root,
            max_hidden_layers=10
        )

    def __createHyperparametersBasedOnModel(self):
        for widget in self.train_model_frame.winfo_children():
            if widget.grid_info().get('row', 0) >= 7:
                widget.destroy()

        if hasattr(self, 'mlp_hyperparameters') and self.mlp_hyperparameters is not None:
            if hasattr(self.mlp_hyperparameters,
                       'hidden_layers_frame') and self.mlp_hyperparameters.hidden_layers_frame is not None:
                self.mlp_hyperparameters.hidden_layers_frame.destroy()
                self.mlp_hyperparameters.hidden_layers_frame = None

        if self.modelOptionMenu.get() == "Random Forest Classifier":
            self.__setRandomForestGui()
        elif self.modelOptionMenu.get() == "MLP Classifier":
            self.__setMlpGui()

    def run(self):
        self.__addHeader()
        self.__createInputData()
        self.__createChooseModel()
        self.__createHyperparametersBasedOnModel()
        self.root.mainloop()


if __name__ == "__main__":
    root = ctk.CTk()
    app = GUITrainModel(root)
    app.run()
