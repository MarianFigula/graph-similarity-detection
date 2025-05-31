import os
import sys
import customtkinter as ctk
from GUI.GUIChooseOptions import GUIChooseOptions

if getattr(sys, 'frozen', False):
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

if __name__ == "__main__":
    root = ctk.CTk()
    app = GUIChooseOptions(root)
    app.run()