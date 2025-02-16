import customtkinter as ctk

from UI.GUIChooseOptions import GUIChooseOptions


if __name__ == "__main__":
    root = ctk.CTk()  # Use CTk instead of Tk for the main window
    app = GUIChooseOptions(root)
    app.run()