import customtkinter as ctk

from GUI.GUIProcessData import GUIProcessData

if __name__ == "__main__":
    root = ctk.CTk()
    app = GUIProcessData(root)
    app.run()