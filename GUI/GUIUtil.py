import customtkinter as ctk
from GUI.NumberInput import NumberInput
from GUI import GUIConstants as guiconst


class GUIUtil:
    fontTitle = ("Lato", 16)

    def __init__(self):
        self.info_window = None

    @staticmethod
    def initialize_window(root):
        for widget in root.winfo_children():
            widget.destroy()

    @staticmethod
    def add_component(self, component_type, frame=None, grid_options=None, **kwargs):
        """
        Adds a component to the window.

        Args:
            component_type (str): The type of component to add (e.g., "Label", "Button", "Entry").
            grid_options (dict): A dictionary containing grid options (e.g., {"row": 0, "column": 0}).
            **kwargs: Additional keyword arguments to configure the component.
        """
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

        if grid_options:
            component.grid(**grid_options)
        else:
            component.pack()

        return component

    @staticmethod
    def create_horizontal_line(frame=None, width=50, height=2, fg_color="gray", **kwargs):
        line = ctk.CTkFrame(frame,width=width, height=height, fg_color=fg_color)
        line.grid(**kwargs)
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
    def reset_label(label_component):
        label_component.configure(text="")

    @staticmethod
    def toggle_component(checkbox_val, component, **kwargs):
        if bool(checkbox_val.get()):
            component.grid(**kwargs)
        else:
            component.grid_remove()

    def __create_info_top_level(self, title, width=300, height=100, **kwargs):
        # TODO: nastavit pozicia kde sa to bude zobrazovat
        info_window = ctk.CTkToplevel()
        info_window.title(title)
        info_window.geometry(f"{width}x{height}")

        info_label = ctk.CTkLabel(info_window,text="", **kwargs)
        info_label.grid(row=0, column=0, padx=30, pady=10)

        info_window.lift()
        info_window.focus_force()

        return info_window

    @staticmethod
    def displayError(frame, text, duration=3000, **kwargs):
        label = ctk.CTkLabel(frame, text=text, text_color=guiconst.COLOR_RED)
        label.grid(**kwargs)

        frame.after(duration, lambda: label.destroy())

        return label

    def create_compare_network_tutorial(self, parent):
        text_box = ctk.CTkTextbox(parent, width=600, height=400)
        text_box.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        text_box._textbox.tag_configure("title", font=("Arial", 16, "bold"), justify="center")
        text_box._textbox.tag_configure("subtitle", font=("Arial", 12, "bold"))
        text_box._textbox.tag_configure("normal", font=("Arial", 11))
        text_box._textbox.tag_configure("tips", font=("Arial", 11, "italic"))

        text_box.insert("end", "NETWORK COMPARISON TUTORIAL\n\n", "title")

        text_box.insert("end", "1. Upload Graphlet Distribution (.csv)\n", "subtitle")
        text_box.insert("end", "   - Click 'Select input file' to upload your first network's graphlet data.\n",
                        "normal")
        text_box.insert("end",
                        "   - Don't have a CSV? Use the 'Process Files' section to generate one from graph files or ORCA outputs.\n\n",
                        "normal")

        text_box.insert("end", "2. Compare Two Networks (Optional)\n", "subtitle")
        text_box.insert("end", "   - Check the box 'Compare between two graphlets' to enable a second file input.\n",
                        "normal")
        text_box.insert("end", "   - Upload the second network's CSV file when prompted.\n\n", "normal")

        text_box.insert("end", "3. Select Machine Learning Model\n", "subtitle")
        text_box.insert("end", "   - Choose from pre-trained models in 'MachineLearningData/saved_models'.\n", "normal")
        text_box.insert("end",
                        "   - You can use your own trained models or those created in the 'Train Model' section.\n\n",
                        "normal")

        text_box.insert("end", "4. Run the Comparison\n", "subtitle")
        text_box.insert("end", "   - Click the 'Compare' button to start analysis.\n", "normal")
        text_box.insert("end", "   - Processing time depends on network size and model complexity.\n\n", "normal")

        text_box.insert("end", "5. View and Save Results\n", "subtitle")
        text_box.insert("end",
                        "   - Button 'Download Results' saves the predictions as files in 'MachineLearningData/predictions'.\n",
                        "normal")
        text_box.insert("end", "   - Button 'Display Results' shows the comparison visually in the application.\n\n", "normal")

        text_box.insert("end", "Tips:\n", "subtitle")
        text_box.insert("end", "- Ensure CSV files follow the correct graphlet format.\n", "tips")
        text_box.insert("end", "- Larger networks may take longer to process.\n", "tips")
        text_box.insert("end", "- Check the documentation for advanced comparison options.\n", "tips")

        text_box.configure(state="disabled")

        return text_box

    def create_train_model_tutorial(self, parent):
        text_box = ctk.CTkTextbox(parent, width=600, height=400)
        text_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        text_box._textbox.tag_configure("title", font=("Arial", 16, "bold"))
        text_box._textbox.tag_configure("subtitle", font=("Arial", 12, "bold"))
        text_box._textbox.tag_configure("normal", font=("Arial", 11))
        text_box._textbox.tag_configure("tips", font=("Arial", 11, "italic"))

        text_box._textbox.tag_configure("center", justify="center")

        text_box.insert("end", "TRAIN MODEL TUTORIAL\n\n", ("title", "center"))

        text_box.insert("end", "1. Prepare Input Files\n", "subtitle")
        text_box.insert("end", "   - You need two CSV files for training:\n", "normal")
        text_box.insert("end", "     a) Graphlet Distribution CSV: Contains graphlet frequencies for each network\n",
                        "normal")
        text_box.insert("end",
                        "     b) Similarity CSV: Contains pairs of networks with similarity labels (Graph1, Graph2, Label)\n\n",
                        "normal")
        text_box.insert("end", "   - Don't have these files? Generate them in the 'Process Files' section.\n\n",
                        "normal")

        text_box.insert("end", "2. Upload Training Data\n", "subtitle")
        text_box.insert("end",
                        "   - Click 'Select graphlet distribution file' to upload your networks' graphlet data.\n",
                        "normal")
        text_box.insert("end", "   - Click 'Select similarity file' to upload your network similarity labels.\n",
                        "normal")
        text_box.insert("end", "   - Ensure your similarity CSV follows the correct format: Graph1, Graph2, Label.\n\n",
                        "normal")

        text_box.insert("end", "3. Select Machine Learning Model\n", "subtitle")
        text_box.insert("end", "   - After uploading both required files, model selection will be enabled.\n", "normal")
        text_box.insert("end", "   - Choose between:\n", "normal")
        text_box.insert("end",
                        "     a) RandomForestClassifier\n", "normal")
        text_box.insert("end", "     b) MLPClassifier\n\n", "normal")

        text_box.insert("end", "4. Train Your Model\n", "subtitle")
        text_box.insert("end", "   - Click the 'Train Model' button to start the training process.\n", "normal")
        text_box.insert("end", "   - Training time depends on dataset size and model complexity.\n\n", "normal")

        text_box.insert("end", "5. Visualize and Save Results\n", "subtitle")
        text_box.insert("end", "   - Button 'Visualize' shows model performance metrics.\n", "normal")
        text_box.insert("end", "   - Button 'Save Model' stores your trained model in 'MachineLearningData/saved_models/'.\n",
                        "normal")
        text_box.insert("end", "   - Saved models can be used in the 'Compare Networks' section for predictions.\n\n",
                        "normal")

        text_box.insert("end", "Tips:\n", "subtitle")
        text_box.insert("end", "- Ensure your training data contains diverse examples for better generalization.\n",
                        "tips")
        text_box.insert("end", "- Larger training datasets typically produce more robust models.\n", "tips")
        text_box.insert("end", "- Try both model types to see which performs better for your specific networks.\n",
                        "tips")
        text_box.insert("end", "- Check model performance metrics before saving to ensure quality.\n", "tips")

        text_box.configure(state="disabled")

        return text_box

    def create_process_files_tutorial(self, parent):
        pass

    def create_tutorial(self, title):
        if self.info_window is None or not self.info_window.winfo_exists():
            self.info_window = self.__create_info_top_level(title, 620, 400)

            if title == "Compare Networks":
                self.create_compare_network_tutorial(self.info_window)
            elif title == "Train Model":
                self.create_train_model_tutorial(self.info_window)
            elif title == "Process Files":
                self.create_process_files_tutorial(self.info_window)

            self.info_window.lift()
            self.info_window.focus_force()
        else:
            self.info_window.lift()
            self.info_window.focus_force()
