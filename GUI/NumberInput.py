import customtkinter as ctk

class NumberInput(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.value = ctk.DoubleVar(value=0.25)

        self.entry = ctk.CTkEntry(self, textvariable=self.value, width=40, justify="center", font=("Lato", 11), state="readonly")
        self.entry.grid(row=0, column=1, padx=0, pady=0)

        self.entry.configure(validate="key", validatecommand=(self.register(self.validate_input), "%P"))

        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=0, column=2, padx=(5, 5), pady=5)

        self.up_button = ctk.CTkButton(self.button_frame, text="▲", width=10, height=10, command=self.increment, font=("Lato", 8))
        self.up_button.pack(pady=(0, 0))

        # Down button to decrease the value
        self.down_button = ctk.CTkButton(self.button_frame, text="▼", width=10, height=10, command=self.decrement, font=("Lato", 8))
        self.down_button.pack(pady=(0, 0))

    def validate_input(self, new_value):
        """Validate the input to ensure it's a number between 0 and 1."""
        try:
            value = float(new_value)
            return 0 <= value <= 1
        except ValueError:
            return False

    def increment(self):
        current_value = self.value.get()
        new_value = min(current_value + 0.1, 1.0)
        self.value.set(round(new_value, 2))

    def decrement(self):
        current_value = self.value.get()
        new_value = max(current_value - 0.1, 0.0)
        self.value.set(round(new_value, 2))

    def get_value(self):
        return self.value.get()

    def set_value(self, value):
        self.value.set(value)

    def setDisabled(self, state):
        self.entry.configure(state="disabled" if state else "normal")
        self.up_button.configure(state="disabled" if state else "normal")
        self.down_button.configure(state="disabled" if state else "normal")
        self.set_value(0.0 if state else 0.25)