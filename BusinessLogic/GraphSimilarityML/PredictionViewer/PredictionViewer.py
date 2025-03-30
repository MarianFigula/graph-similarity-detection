from tkinter import ttk
import customtkinter as ctk

class PredictionViewer:
    def __init__(self, prediction_result_df):
        self.prediction_result_df = prediction_result_df

    def showPredictions(self):
        top_level = ctk.CTkToplevel()
        top_level.title("Prediction Results")
        top_level.geometry("800x400")

        frame = ctk.CTkFrame(top_level)
        frame.pack(expand=True, fill="both", padx=10, pady=10)

        columns = list(self.prediction_result_df.columns)
        tree = ttk.Treeview(frame, columns=columns, show="headings")

        for col in columns:
            tree.heading(col, text=col)

        for _, row in self.prediction_result_df.iterrows():
            tree.insert("", "end", values=list(row))

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        tree.pack(expand=True, fill="both")

        top_level.lift()
        top_level.focus_force()