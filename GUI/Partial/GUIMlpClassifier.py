from tkinter import IntVar
import customtkinter as ctk
import GUI.GUIConstants as guiconst


class GUIMlpClassifier:
    """
    A reusable class for creating hyperparameter configuration frames for neural networks.
    This class creates a main frame for hyperparameter inputs and a separate frame for
    configuring hidden layers, designed to be placed side by side in the parent container.
    """

    def __init__(self, parent, gui_util, root, max_hidden_layers=6):
        """
        Initialize the hyperparameter frame

        Parameters:
        -----------
        parent : tk.Frame or tk.Toplevel
            The parent container where this frame will be placed
        gui_util : GuiUtil
            Utility class for creating GUI components
        root : Tk
            The root window containing fonts and other global resources
        max_hidden_layers : int
            Maximum number of hidden layers that can be configured
        """
        self.parent = parent
        self.guiUtil = gui_util
        self.root = root
        self.max_hidden_layers = max_hidden_layers

        self.main_frame = ctk.CTkFrame(parent, width=600, height=400)
        self.main_frame.grid(row=7, column=0, columnspan=2, padx=(40, 20), pady=10, sticky="nsew")

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_propagate(False)

        self.hidden_layers_frame = ctk.CTkFrame(self.root, width=150, height=350)
        self.hidden_layers_frame.grid(row=2, column=1, padx=(0, 40), pady=240, sticky="nsew")

        self._create_hyperparameters()
        self._create_hidden_layer_inputs()

        self.hidden_layer_inputs = []

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.hidden_layers_frame.grid_columnconfigure(0, weight=1)
        self.hidden_layers_frame.grid_columnconfigure(1, weight=1)
        self.hidden_layers_frame.grid_propagate(False)

    def grid(self, main_grid_options, hidden_grid_options):
        """
        Position both frames in the parent container

        Parameters:
        -----------
        main_grid_options : dict
            Grid options for the main hyperparameters frame
        hidden_grid_options : dict
            Grid options for the hidden layers frame
        """
        self.main_frame.grid(**main_grid_options)
        self.hidden_layers_frame.grid(**hidden_grid_options)

    def _create_hyperparameters(self):
        """
        Create all hyperparameter input fields and controls in the main frame

        Hyperparameters:
        - Number of hidden layers
        - Number of epochs
        - Batch size
        - Learning rate
        - Early stopping and patience
        - Training buttons
        - Visualization options
        - Save options
        """
        # Title for the frame
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Model Hyperparameters",
            grid_options={"row": 0, "column": 0, "columnspan": 2, "sticky": "w", "padx": 30, "pady": (5, 0)},
            font=self.root.fontMiddle
        )

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Number of hidden layers",
            grid_options={"row": 1, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.font
        )

        self.num_hidden_layers = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self.main_frame,
            grid_options={"row": 1, "column": 1, "sticky": "e", "padx": 30, "pady": 5},
            min_value=1,
            max_value=self.max_hidden_layers,
            default_value=1,
            step=1,
            data_type=int,
            command=lambda: self._update_hidden_layers()
        )

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Number of epochs",
            grid_options={"row": 2, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.font
        )

        self.num_epochs = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self.main_frame,
            grid_options={"row": 2, "column": 0, "sticky": "e", "padx": 30, "pady": 5},
            min_value=1,
            max_value=1000,
            default_value=100,
            step=1,
            data_type=int
        )

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Batch size",
            grid_options={"row": 2, "column": 1, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.font
        )

        self.batch_size = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self.main_frame,
            grid_options={"row": 2, "column": 1, "sticky": "e", "padx": 30, "pady": 5},
            min_value=1,
            max_value=256,
            default_value=32,
            step=1,
            data_type=int
        )

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Learning rate",
            grid_options={"row": 3, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.font
        )

        self.learning_rate = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self.main_frame,
            grid_options={"row": 3, "column": 0, "sticky": "e", "padx": 30, "pady": 5},
            min_value=0.0,
            max_value=1.0,
            default_value=0.001,
            step=0.0001,
            data_type=float
        )

        self.early_stopping_var = IntVar()
        self.early_stopping = self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.main_frame,
            text="Early stopping",
            variable=self.early_stopping_var,
            grid_options={"row": 4, "column": 0, "sticky": "w", "padx": 30, "pady": 5},
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            font=self.root.font,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN,
            border_width=2,
            command=self._toggle_patience
        )

        self.patience_label = self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.main_frame,
            text="Patience",
            grid_options={"row": 4, "column": 1, "sticky": "w", "padx": 30, "pady": 5},
            font=self.root.font
        )
        self.patience_label.grid_remove()

        self.patience = self.guiUtil.add_component(
            self,
            component_type="NumberInput",
            frame=self.main_frame,
            grid_options={"row": 4, "column": 1, "sticky": "e", "padx": 30, "pady": 5},
            min_value=1,
            max_value=100,
            default_value=10,
            step=1,
            data_type=int
        )
        self.patience.grid_remove()

        self.train_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.main_frame,
            text="Train Model",
            grid_options={"row": 5, "column": 0, "columnspan": 2, "sticky": "n", "padx": 10, "pady": 5},
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            width=180,
            height=25,
            command=self._train_model
        )

        # Horizontal separator
        self.guiUtil.create_horizontal_line(
            self.main_frame,
            width=380,  # Slightly smaller than frame width
            row=6,
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
            grid_options={"row": 7, "column": 0, "sticky": "w", "padx": (30, 0), "pady": 5},
            font=self.root.fontMiddle
        )

        # Visualization options checkboxes in a 2x2 grid
        self.accuracy_loss_curves_var = IntVar()
        self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.main_frame,
            text="Accuracy/Loss Curves",
            variable=self.accuracy_loss_curves_var,
            grid_options={"row": 8, "column": 0, "sticky": "w", "padx": (30, 0), "pady": 5},
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
            grid_options={"row": 8, "column": 0, "sticky": "e", "padx": (30, 0), "pady": 5},
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
            grid_options={"row": 8, "column": 1, "sticky": "w", "padx": (30, 0), "pady": 5},
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
            grid_options={"row": 8, "column": 1, "sticky": "e", "padx": 30, "pady": 5},
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
            grid_options={"row": 9, "column": 0, "sticky": "n", "padx": 10, "pady": (5, 10)},
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            width=180,
            height=25,
            command=self._visualize_results
        )

        self.save_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.main_frame,
            text="Save Model",
            grid_options={"row": 9, "column": 1, "sticky": "n", "padx": 10, "pady": (5, 10)},
            width=180,
            height=25,
            command=self._save_model
        )

    def _create_hidden_layer_inputs(self):
        """Create input fields for hidden layers in the hidden_layers_frame"""
        # Clear any existing widgets in hidden_layers_frame
        for widget in self.hidden_layers_frame.winfo_children():
            widget.destroy()

        # Reset hidden layer inputs list
        self.hidden_layer_inputs = []

        # Title for the frame
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.hidden_layers_frame,
            text="Hidden Layer Configuration",
            grid_options={"row": 0, "column": 0, "columnspan": 2, "sticky": "w", "padx": 20, "pady": 10},
            font=self.root.fontMiddle
        )

        # Get the number of hidden layers
        num_layers = 1
        if hasattr(self, 'num_hidden_layers'):
            try:
                num_layers = self.num_hidden_layers.get()
            except:
                num_layers = 1

        print(num_layers)
        for i in range(num_layers):
            neuron_label = self.guiUtil.add_component(
                self,
                component_type="Label",
                frame=self.hidden_layers_frame,
                text=f"Layer {1 + i}: Neurons",
                grid_options={"row": 1 + i, "column": 0, "sticky": "w", "padx": 5, "pady": 2},
                font=self.root.font
            )

            neurons = self.guiUtil.add_component(
                self,
                component_type="NumberInput",
                frame=self.hidden_layers_frame,
                grid_options={"row": 1 + i, "column": 0, "sticky": "e", "padx": 5, "pady": 2},
                min_value=1,
                max_value=1000,
                default_value=100,
                step=1,
                data_type=int
            )

            dropout_label = self.guiUtil.add_component(
                self,
                component_type="Label",
                frame=self.hidden_layers_frame,
                text=f"Dropout rate:",
                grid_options={"row": 1 + i, "column": 1, "sticky": "w", "padx": 5, "pady": 2},
                font=self.root.font
            )

            dropout = self.guiUtil.add_component(
                self,
                component_type="NumberInput",
                frame=self.hidden_layers_frame,
                grid_options={"row": 1 + i, "column": 1, "sticky": "e", "padx": 5, "pady": 2},
                min_value=0.0,
                max_value=1.0,
                default_value=0.0,
                step=0.1,
                data_type=float
            )

            self.hidden_layer_inputs.append({
                'neuron_label': neuron_label,
                'neurons': neurons,
                'dropout_label': dropout_label,
                'dropout': dropout
            })

    def _update_hidden_layers(self):
        """
        Update the hidden layers frame based on the selected number of layers
        This function is called when the number of hidden layers is changed
        """
        # Re-create all hidden layer inputs
        self._create_hidden_layer_inputs()

    def _toggle_patience(self):
        """Show or hide patience input based on early stopping checkbox"""
        if self.early_stopping_var.get() == 1:
            self.patience_label.grid()
            self.patience.grid()
        else:
            self.patience_label.grid_remove()
            self.patience.grid_remove()

    def _train_model(self):
        """Collect hyperparameters and train the model"""
        # Placeholder for model training logic
        hyperparams = self.get_hyperparameters()
        print("Training model with hyperparameters:", hyperparams)
        # Add actual training logic here

    def _visualize_results(self):
        """Visualize the training results based on selected options"""
        # Placeholder for visualization logic
        options = {
            'accuracy_loss': self.accuracy_loss_curves_var.get() == 1,
            'confusion_matrix': self.confusion_matrix_var.get() == 1,
            'roc_curve': self.roc_curve_var.get() == 1,
            'classification_report': self.classification_report_var.get() == 1
        }
        print("Visualizing with options:", options)
        # Add actual visualization logic here

    def _save_model(self):
        """Save the trained model to a file"""
        # Placeholder for model saving logic
        print("Saving model...")
        # Add actual saving logic here

    def get_hyperparameters(self):
        """
        Collect and return all hyperparameters as a dictionary

        Returns:
        --------
        dict : Dictionary containing all hyperparameter values
        """
        hidden_layers = []
        for i, layer in enumerate(self.hidden_layer_inputs):
            if i < self.num_hidden_layers.get():
                hidden_layers.append({
                    'neurons': layer['neurons'].get(),
                    'dropout': layer['dropout'].get()
                })

        params = {
            'num_hidden_layers': self.num_hidden_layers.get(),
            'hidden_layers': hidden_layers,
            'num_epochs': self.num_epochs.get(),
            'batch_size': self.batch_size.get(),
            'learning_rate': self.learning_rate.get(),
            'early_stopping': self.early_stopping_var.get() == 1
        }

        if params['early_stopping']:
            params['patience'] = self.patience.get()

        return params

    def set_hyperparameters(self, params):
        """
        Set hyperparameter values from a dictionary

        Parameters:
        -----------
        hyperparams : dict
            Dictionary containing hyperparameter values
        """
        if 'num_hidden_layers' in params:
            self.num_hidden_layers.set(params['num_hidden_layers'])
            self._update_hidden_layers()

            if 'hidden_layers' in params:
                for i, layer in enumerate(params['hidden_layers']):
                    if i < len(self.hidden_layer_inputs):
                        if 'neurons' in layer:
                            self.hidden_layer_inputs[i]['neurons'].set_value(layer['neurons'])
                        if 'dropout' in layer:
                            self.hidden_layer_inputs[i]['dropout'].set_value(layer['dropout'])

        if 'num_epochs' in params:
            self.num_epochs.set_value(params['num_epochs'])

        if 'batch_size' in params:
            self.batch_size.set_value(params['batch_size'])

        if 'learning_rate' in params:
            self.learning_rate.set_value(params['learning_rate'])

        if 'early_stopping' in params:
            self.early_stopping_var.set(1 if params['early_stopping'] else 0)
            self._toggle_patience()

            if params['early_stopping'] and 'patience' in params:
                self.patience.set_value(params['patience'])
