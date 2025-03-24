import customtkinter as ctk


class NumberInput(ctk.CTkFrame):
    def __init__(self, master, min_value=0.0, max_value=1.0, step=0.1,
                 default_value=0.25, disabled_value=0.0, data_type=float, **kwargs):

        super().__init__(master, **kwargs)

        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.default_value = default_value
        self.disabled_value = disabled_value
        self.data_type = data_type

        self.min_value = self.data_type(self.min_value)
        self.max_value = self.data_type(self.max_value)
        self.step = self.data_type(self.step)
        self.default_value = self.data_type(self.default_value)
        self.disabled_value = self.data_type(self.disabled_value)

        self.value = ctk.DoubleVar(value=self.default_value)

        self.entry = ctk.CTkEntry(self, textvariable=self.value, width=50, justify="center", font=("Lato", 11))
        self.entry.grid(row=0, column=1, padx=0, pady=0)

        self.entry.configure(validate="key", validatecommand=(self.register(self.validate_input), "%P"))

        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=0, column=2, padx=(5, 5), pady=5)

        self.up_button = ctk.CTkButton(self.button_frame, text="▲", width=10, height=10, command=self.increment, font=("Lato", 8))
        self.up_button.pack(pady=(0, 0))

        self.down_button = ctk.CTkButton(self.button_frame, text="▼", width=10, height=10, command=self.decrement, font=("Lato", 8))
        self.down_button.pack(pady=(0, 0))

    def validate_input(self, new_value):
        """Validate the input to ensure it's a number between min_value and max_value."""
        try:
            value = self.data_type(new_value)  # Convert to the correct data type
            return self.min_value <= value <= self.max_value
        except ValueError:
            return False

    def increment(self):
        current_value = self.data_type(self.value.get())  # Convert to the correct data type
        new_value = min(current_value + self.step, self.max_value)
        self.value.set(self.data_type(new_value))

    def decrement(self):
        current_value = self.data_type(self.value.get())
        new_value = max(current_value - self.step, self.min_value)
        self.value.set(self.data_type(new_value))

    def get_value(self):
        return self.data_type(self.value.get())

    def set_value(self, value):
        value = self.data_type(value)
        if self.min_value <= value <= self.max_value:
            self.value.set(value)
        else:
            raise ValueError(f"Value must be between {self.min_value} and {self.max_value}")

    def setDisabled(self, state):
        self.entry.configure(state="disabled" if state else "normal")
        self.up_button.configure(state="disabled" if state else "normal")
        self.down_button.configure(state="disabled" if state else "normal")
        self.set_value(self.disabled_value if state else self.default_value)

    def set_min_value(self, min_value):
        self.min_value = self.data_type(min_value)

    def set_max_value(self, max_value):
        self.max_value = self.data_type(max_value)

    def set_step(self, step):
        self.step = self.data_type(step)

    def set_default_value(self, default_value):
        self.default_value = self.data_type(default_value)

    def set_disabled_value(self, disabled_value):
        self.disabled_value = self.data_type(disabled_value)

    def set_data_type(self, data_type):
        """Set the data type (int or float) dynamically."""
        if data_type not in (int, float):
            raise ValueError("data_type must be either int or float")
        self.data_type = data_type

        self.min_value = self.data_type(self.min_value)
        self.max_value = self.data_type(self.max_value)
        self.step = self.data_type(self.step)
        self.default_value = self.data_type(self.default_value)
        self.disabled_value = self.data_type(self.disabled_value)
        self.set_value(self.data_type(self.value.get()))