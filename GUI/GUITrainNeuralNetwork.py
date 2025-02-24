import os
from tkinter import filedialog, IntVar
import customtkinter as ctk

from BusinessLogic.DataVisualiser.DataVisualiser import DataVisualiser
from BusinessLogic.ProcessFiles.SimilarityHandler import SimilarityHandler
from BusinessLogic.ProcessFiles.SnapShotOfGraphletsAsGraph import SnapShotOfGraphletsAsGraph
from GUI.GUIUtil import GUIUtil
from BusinessLogic.ProcessFiles.ProcessInAndOutFiles import ProcessInAndOutFiles
import GUI.GUIConstants as guiconst



class GUITrainNeuralNetwork:
    def __init__(self, root):
        self.root = root
        self.root.title("Train neural network")
        self.root.geometry("1000x700")
        self.root.fontTitle = ("Lato", 16)
        self.root.font = ("Lato", 12)
        self.root.smallFont = ("Lato", 10)
        self.guiUtil = GUIUtil()
        self.process_files = None
        self.create_snapshots = None
        self.orbit_counts_df = None
        self.similarityHandler = None
        self.similarity_measures = None
        self.root.grid_columnconfigure(1, weight=1)  # Make the column expand to center content
        self.root.grid_columnconfigure(2, weight=1)  # Make the column expand to center content
        self.root.grid_columnconfigure(3, weight=1)  # Make the column expand to center content
        self.root.grid_columnconfigure(4, weight=1)  # Make the column expand to center content

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        # Create a main frame to hold all components
        self.process_data_frame = ctk.CTkFrame(self.root, width=300, height=300)
        self.process_data_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ns")  # No horizontal expansion

    def __goBackToOptions(self):
        self.guiUtil.removeWindow(root=self.root)
        # Create the new GUI in the same root window
        from GUI.GUIChooseOptions import GUIChooseOptions
        app = GUIChooseOptions(self.root)
        app.run()

    def __addHeader(self):
        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.root,
            text="< Back to options",
            command=lambda: self.__goBackToOptions(),
            grid_options={"row": 0, "column": 0, "columnspan": 1, "sticky": "n", "pady": 10},
            font=self.root.font,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            width=30,
            height=25

        )
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.root,
            text="Train Neural Network",
            grid_options={"row": 0, "column": 0, "columnspan": 7, "sticky": "n", "pady": 5},
            font=self.root.fontTitle
        )

    def __handleSelectDirectory(self, entry, button=None):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, ctk.END)
            entry.insert(ctk.END, path)

            if button:
                button.configure(state="normal")

    def __handleRemoveDirectory(self, entry, button=None):
        entry.delete(0, ctk.END)
        if button:
            button.configure(state="disabled")

    def __handleProcessFiles(self, input_folder_path, output_folder_path, is_out_files=False, should_create_images=False):
        # self.progress_bar.start()
        print("input_folder_path:", input_folder_path)
        print("output_folder_path:", output_folder_path)
        print("is_out_files:", is_out_files)
        print("Current working directory:", os.getcwd())

        self.process_files = ProcessInAndOutFiles(
            input_folder_path=input_folder_path,
            output_folder_path=output_folder_path,
            is_out_files=is_out_files)

        if not self.process_files.process():
            return

        if should_create_images:
            self.create_snapshots = SnapShotOfGraphletsAsGraph(self.process_files.get_orbit_counts_df())
            self.create_snapshots.create_images()

        self.show_graphs_button.configure(state="normal")

        self.__handleCheckboxLabelingMethodStates()
        self.count_similarities_button.configure(state="normal")
        # self.progress_bar.set(100)
        # self.progress_bar.stop()

    def disableCheckboxWithValue(self, checkbox, checkbox_val):
        checkbox.configure(state="disabled")
        checkbox_val.set(0)

    def enableCheckboxWithValue(self, checkbox, checkbox_val):
        checkbox.configure(state="normal")
        checkbox_val.set(1)

    def toggleCheckboxWithValue(self, checkbox, checkbox_val):
        checkbox.configure(state="normal" if bool(checkbox_val.get()) else "disabled")
        checkbox_val.set(1 if not bool(checkbox_val.get()) else 0)

    def __handleCheckboxLabelingMethodStates(self):
        # if not bool(self.out_files_val.get()):
        #     self.enableCheckboxWithValue(self.netsimile_checkbox, self.netsimile_val)
        # else:
            # self.disableCheckboxWithValue(self.netsimile_checkbox, self.netsimile_val)

        if bool(self.create_images_val.get()):
            self.enableCheckboxWithValue(self.resnet_checkbox, self.resnet_val)
        else:
            self.disableCheckboxWithValue(self.resnet_checkbox, self.resnet_val)

        self.enableCheckboxWithValue(self.hellinger_checkbox, self.hellinger_val)

    def __handleComputeSimilarity(self):
        orbit_counts_df = self.process_files.get_orbit_counts_df()
        self.similarityHandler = SimilarityHandler(orbit_counts_df,
                                                   self.input_entry.get(),
                                                   self.create_snapshots.getImgDir()
                                                   )

        self.similarity_measures = self.similarityHandler.countSimilarities(
            hellinger_check_val=bool(self.hellinger_val.get()),
            netsimile_check_val=bool(self.netsimile_val.get()),
            resnet_check_val=bool(self.resnet_val.get()))

        # self.exportSimilarityMeasures()
        self.label_similarity_button.configure(state="normal")

    def __handleLabelSimilarities(self):
        self.similarityHandler.labelSimilarities(
            hellinger_check_val=bool(self.hellinger_val.get()),
            netsimile_check_val=bool(self.netsimile_val.get()),
            resnet_check_val=bool(self.resnet_val.get())
        )

        self.exportSimilarityMeasures()



    def exportSimilarityMeasures(self):
        self.similarityHandler.exportSimilarity(self.output_entry.get() + "/similarity_measures.csv")

    def __createProcessingDataFrame(self):
        """Input"""

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.process_data_frame,
            text="Input directory",
            grid_options={"row": 1, "column": 0},
            font=self.root.font,
            anchor="center"
        )

        self.input_entry = self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.process_data_frame,
            grid_options={"row": 2, "column": 0, "sticky": "w", "padx": 10},
            font=self.root.font,
            width=125,
            height=20
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Select input directory",
            grid_options={"row": 3, "column": 0, "sticky": "w", "padx": 10, "pady": 5},
            font=self.root.font,
            width=50,
            command=lambda: self.__handleSelectDirectory(self.input_entry, self.process_files_button)
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Reset input directory",
            grid_options={"row": 4, "column": 0, "sticky": "w", "padx": 10},
            font=self.root.font,
            width=50,
            fg_color=guiconst.COLOR_RED,
            hover_color=guiconst.COLOR_RED_HOVER,
            command=lambda: self.__handleRemoveDirectory(self.input_entry, self.process_files_button)
        )

        """Output"""

        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.process_data_frame,
            text="Output directory",
            grid_options={"row": 1, "column": 1, "sticky": "w", "padx": 20, "pady": 5},
            font=self.root.font,
            anchor="w"
        )

        self.output_entry = self.guiUtil.add_component(
            self,
            component_type="Entry",
            frame=self.process_data_frame,
            grid_options={"row": 2, "column": 1, "sticky": "w", "padx": (0, 10), "pady": 5},
            font=self.root.font,
            width=130,
            height=20
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Select output directory",
            grid_options={"row": 3, "column": 1, "sticky": "w", "pady": 5},
            font=self.root.font,
            width=50,
            command=lambda: self.__handleSelectDirectory(self.output_entry)
        )

        self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Reset output directory",
            grid_options={"row": 4, "column": 1, "sticky": "w"},
            font=self.root.font,
            width=50,
            fg_color=guiconst.COLOR_RED,
            hover_color=guiconst.COLOR_RED_HOVER,
            command=lambda: self.__handleRemoveDirectory(self.output_entry)
        )

        """Is Orca files checkbox"""

        self.out_files_val = IntVar()
        self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="Data are Orca files",
            grid_options={"row": 5, "column": 0, "sticky": "w", "padx": 10, "pady": (15, 0)},
            variable=self.out_files_val,
            font=self.root.font,
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2
        )

        self.create_images_val = IntVar()
        self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="Create images",
            grid_options={"row": 5, "column": 1, "sticky": "w", "padx": 10, "pady": (15, 0)},
            variable=self.create_images_val,
            font=self.root.font,
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2
        )

        self.process_files_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Process files",
            grid_options={"row": 6, "column": 0, "sticky": "ew", "padx": 10, "pady": (15, 0)},
            font=self.root.font,
            width=50,
            height=25,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="disabled",
            command=lambda: self.__handleProcessFiles(
                self.input_entry.get(),
                self.output_entry.get(),
                bool(self.out_files_val.get()),
                bool(self.create_images_val.get())
            )
        )

        self.show_graphs_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            text="Show graphs",
            frame=self.process_data_frame,
            grid_options={"row": 6, "column": 1, "sticky": "ew", "padx": (5,10), "pady": (15, 0)},
            font=self.root.font,
            width=50,
            height=25,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="disabled",
            command=lambda: DataVisualiser(self.process_files.get_orbit_counts_df()).visualize()
        )

        # self.progress_bar = self.guiUtil.add_component(
        #     self,
        #     component_type="Progressbar",
        #     frame=self.process_data_frame,
        #     grid_options={"row": 7, "column": 0, "columnspan": 2, "sticky": "ew", "padx": 10, "pady": (15, 0)}
        # )
        # self.progress_bar.set(0)

    def __createChooseLabelingMethods(self):
        self.guiUtil.add_component(
            self,
            component_type="Label",
            frame=self.process_data_frame,
            text="Choose labeling methods",
            grid_options={"row": 8, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font
        )

        self.hellinger_val = IntVar()
        self.hellinger_checkbox = self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="Hellinger",
            grid_options={"row": 9, "column": 0, "sticky": "w", "padx": (10,0)},
            variable=self.hellinger_val,
            font=self.root.font,
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2,
            state="disabled"
        )

        self.netsimile_val = IntVar()
        self.netsimile_checkbox = self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="NetSimile (input data must be graphs)",
            grid_options={"row": 10, "column": 0, "columnspan": 2, "sticky": "w", "padx": (10, 0)},
            variable=self.netsimile_val,
            font=self.root.font,
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2,
            state="normal"
        )

        self.resnet_val = IntVar()
        self.resnet_checkbox = self.guiUtil.add_component(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="Resnet (images has to be created)",
            grid_options={"row": 11, "column": 0, "columnspan": 2, "sticky": "w", "padx": 10},
            variable=self.resnet_val,
            font=self.root.font,
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2,
            state="disabled"
        )

        self.count_similarities_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            text="Count similarities",
            frame=self.process_data_frame,
            grid_options={"row": 12, "column": 0, "columnspan": 2, "sticky": "ew", "padx": 10, "pady": (15, 0)},
            font=self.root.font,
            width=50,
            height=25,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="disabled",
            command=lambda: self.__handleComputeSimilarity()
        )

        self.label_similarity_button = self.guiUtil.add_component(
            self,
            component_type="Button",
            text="Label similarities",
            frame=self.process_data_frame,
            grid_options={"row": 13, "column": 0, "columnspan": 2, "sticky": "ew", "padx": 10, "pady": (15, 0)},
            font=self.root.font,
            width=50,
            height=25,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="disabled",
            command=lambda: self.__handleLabelSimilarities()
        )


    def run(self):
        self.__addHeader()
        self.__createProcessingDataFrame()
        self.guiUtil.create_horizontal_line(
            self.process_data_frame,
            width=300,
            row=7,
            column=0,
            columnspan=3,
            padx=5,
            pady=15,
            sticky="w"
        )
        self.__createChooseLabelingMethods()
        self.root.mainloop()


if __name__ == "__main__":
    root = ctk.CTk()  # Use CTk instead of Tk for the main window
    app = GUITrainNeuralNetwork(root)
    app.run()