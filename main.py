import customtkinter as ctk

from GUI.GUIChooseOptions import GUIChooseOptions

if __name__ == "__main__":
    root = ctk.CTk()
    app = GUIChooseOptions(root)
    app.run()