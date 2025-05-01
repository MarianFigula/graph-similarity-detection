import os
from tkinter import filedialog, IntVar
import customtkinter as ctk

from BusinessLogic.DataVisualiser.DataVisualiser import DataVisualiser
from BusinessLogic.Exception.EmptyDataException import EmptyDataException
from BusinessLogic.Exception.WeightSumException import WeightSumException
from BusinessLogic.ProcessFiles.SimilarityHandler import SimilarityHandler
from BusinessLogic.ProcessFiles.SnapShotOfGraphletsAsGraph import SnapShotOfGraphletsAsGraph
from GUI.GUIUtil import GUIUtil
from BusinessLogic.ProcessFiles.ProcessInAndOutFiles import ProcessInAndOutFiles
import GUI.GUIConstants as guiconst


class GUIProcessData:
    def __init__(self, root):
        self.root = root
        self.root.title("Data processing")

        self.root.fontTitle = ("Lato", 16)
        self.root.font = ("Lato", 12)
        self.root.smallFont = ("Lato", 10)
        self.guiUtil = GUIUtil()
        self.process_files = None
        self.create_snapshots = None
        self.similarityHandler = None
        self.similarity_measures = None

        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()

        w = 470
        h = 700
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)

        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)
        self.root.grid_columnconfigure(4, weight=1)

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.process_data_frame_width = w - 2 * 40

        self.process_data_frame = ctk.CTkFrame(self.root, width=self.process_data_frame_width, height=1200)
        self.process_data_frame.grid(row=2, column=0, padx=40, pady=10, sticky="ns")

    def __goBackToOptions(self):
        self.guiUtil.removeWindow(root=self.root)
        from GUI.GUIChooseOptions import GUIChooseOptions
        app = GUIChooseOptions(self.root)
        app.run()

    def __addHeader(self):
        self.guiUtil.addComponent(
            self,
            component_type="Button",
            frame=self.root,
            text="< Back to options",
            command=lambda: self.__goBackToOptions(),
            grid_options={"row": 0, "column": 0, "sticky": "nw", "pady": 10, "padx": 15},
            font=self.root.font,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            width=30,
            height=25

        )

        self.guiUtil.addComponent(
            self,
            component_type="Button",
            text="?",
            grid_options={"row": 0, "column": 0, "sticky": "ne", "pady": 10, "padx": 20},
            font=self.root.font,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            width=30,
            height=25,
            command=lambda: self.guiUtil.createTutorial("Process Files"),
        )

        self.guiUtil.addComponent(
            self,
            component_type="Label",
            frame=self.root,
            text="Process Data",
            grid_options={"row": 1, "column": 0, "columnspan": 7, "sticky": "ew", "pady": 5},
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

    def __handleProcessFiles(self, input_folder_path, output_folder_path, is_out_files=False,
                             should_create_images=False):
        error = ""
        print("input_folder_path:", input_folder_path)
        print("output_folder_path:", output_folder_path)
        print("is_out_files:", is_out_files)
        print("Current working directory:", os.getcwd())
        try:
            self.process_files = ProcessInAndOutFiles(
                input_folder_path=input_folder_path,
                output_folder_path=output_folder_path,
                is_out_files=is_out_files)

            if not self.process_files.process():
                self.guiUtil.displayError(self.process_data_frame, "Error processing files")
                return

            if should_create_images:
                self.create_snapshots = SnapShotOfGraphletsAsGraph(self.process_files.get_orbit_counts_df(),
                                                                   input_folder_path, output_folder_path)
                self.create_snapshots.create_images()

            self.show_graphs_button.configure(state="normal")

            self.__handleCheckboxLabelingMethodStates()
            self.count_similarities_button.configure(state="normal")

            self.process_files_complete_label.configure(
                text=f"Graphlet counts saved in\n{self.process_files.graphlet_counts_filename}!")
            self.process_data_frame.after(2000, lambda: self.guiUtil.resetLabel(self.process_files_complete_label))

        except EmptyDataException as e:
            error = str(e)
        except Exception as e:
            error = str(e)
        finally:
            if error != "":
                self.guiUtil.displayError(self.process_data_frame, error, row=16, column=0, columnspan=2)

    def __disableCheckboxWithValue(self, checkbox, checkbox_val):
        checkbox.configure(state="disabled")
        checkbox_val.set(0)

    def __enableCheckboxWithValue(self, checkbox, checkbox_val):
        checkbox.configure(state="normal")
        checkbox_val.set(1)

    def __handleCheckboxLabelingMethodStates(self):
        if bool(self.create_images_val.get()):
            self.__enableCheckboxWithValue(self.resnet_checkbox, self.resnet_val)
            self.resnet_weight.setDisabled(False)
        else:
            self.__disableCheckboxWithValue(self.resnet_checkbox, self.resnet_val)
            self.resnet_weight.setDisabled(True)

        if bool(self.out_files_val.get()):
            self.__disableCheckboxWithValue(self.netsimile_checkbox, self.netsimile_val)
            self.netsimile_weight.setDisabled(True)
        else:
            self.__enableCheckboxWithValue(self.netsimile_checkbox, self.netsimile_val)
            self.netsimile_weight.setDisabled(False)

        self.__enableCheckboxWithValue(self.hellinger_checkbox, self.hellinger_val)
        self.hellinger_weight.setDisabled(not self.hellinger_val.get())

        self.__enableCheckboxWithValue(self.kstest_checkbox, self.kstest_val)
        self.kstest_weight.setDisabled(not self.kstest_val.get())

    def __handleComputeSimilarity(self):
        orbit_counts_df = self.process_files.get_orbit_counts_df()
        self.similarityHandler = SimilarityHandler(orbit_counts_df,
                                                   self.input_entry.get(),
                                                   self.create_snapshots.getImgDir() if self.create_snapshots else None
                                                   )

        self.similarity_measures = self.similarityHandler.countSimilarities(
            hellinger_check_val=bool(self.hellinger_val.get()),
            netsimile_check_val=bool(self.netsimile_val.get()),
            resnet_check_val=bool(self.resnet_val.get()),
            ks_check_val=bool(self.kstest_val.get())
        )

        self.label_similarity_button.configure(state="normal")
        self.exportSimilarityMeasures()

    def __handleLabelSimilarities(self):
        error = ""
        try:
            self.similarityHandler.labelSimilarities(
                hellinger_check_val=bool(self.hellinger_val.get()),
                netsimile_check_val=bool(self.netsimile_val.get()),
                resnet_check_val=bool(self.resnet_val.get()),
                ks_check_val=bool(self.kstest_val.get()),
                hellinger_weight=float(self.hellinger_weight.get()),
                netsimile_weight=float(self.netsimile_weight.get()),
                resnet_weight=float(self.resnet_weight.get()),
                ks_weight=float(self.kstest_weight.get())
            )
            self.exportSimilarityMeasures()

        except WeightSumException as e:
            error = str(e)
        except Exception as e:
            error = str(e)
        finally:
            if error != "":
                self.guiUtil.displayError(self.process_data_frame, error, row=16, column=0, columnspan=2)

    def exportSimilarityMeasures(self):
        self.similarityHandler.exportSimilarity(self.output_entry.get() + "/similarity_measures.csv")

    def __createProcessingDataFrame(self):
        """Input"""

        self.guiUtil.addComponent(
            self,
            component_type="Label",
            frame=self.process_data_frame,
            text="Input directory",
            grid_options={"row": 2, "column": 0, "sticky": "w", "padx": 35},
            font=self.root.font,
            anchor="center"
        )

        self.input_entry = self.guiUtil.addComponent(
            self,
            component_type="Entry",
            frame=self.process_data_frame,
            grid_options={"row": 3, "column": 0, "sticky": "w", "padx": 10},
            font=self.root.font,
            width=125,
            height=20
        )

        self.guiUtil.addComponent(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Select input directory",
            grid_options={"row": 4, "column": 0, "sticky": "w", "padx": 10, "pady": 5},
            font=self.root.font,
            width=50,
            command=lambda: self.__handleSelectDirectory(self.input_entry, self.process_files_button)
        )

        self.guiUtil.addComponent(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Reset input directory",
            grid_options={"row": 5, "column": 0, "sticky": "w", "padx": 10},
            font=self.root.font,
            width=50,
            fg_color=guiconst.COLOR_RED,
            hover_color=guiconst.COLOR_RED_HOVER,
            command=lambda: self.__handleRemoveDirectory(self.input_entry, self.process_files_button)
        )

        """Output"""

        self.guiUtil.addComponent(
            self,
            component_type="Label",
            frame=self.process_data_frame,
            text="Output directory",
            grid_options={"row": 2, "column": 1, "sticky": "w", "padx": (30, 0), "pady": 5},
            font=self.root.font,
            anchor="w"
        )

        self.output_entry = self.guiUtil.addComponent(
            self,
            component_type="Entry",
            frame=self.process_data_frame,
            grid_options={"row": 3, "column": 1, "sticky": "w", "padx": (10, 10), "pady": 5},
            font=self.root.font,
            width=130,
            height=20
        )

        self.guiUtil.addComponent(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Select output directory",
            grid_options={"row": 4, "column": 1, "sticky": "w", "padx": 10, "pady": 5},
            font=self.root.font,
            width=50,
            command=lambda: self.__handleSelectDirectory(self.output_entry)
        )

        self.guiUtil.addComponent(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Reset output directory",
            grid_options={"row": 5, "column": 1, "sticky": "w", "padx": 10},
            font=self.root.font,
            width=50,
            fg_color=guiconst.COLOR_RED,
            hover_color=guiconst.COLOR_RED_HOVER,
            command=lambda: self.__handleRemoveDirectory(self.output_entry)
        )

        """Is Orca files checkbox"""

        self.out_files_val = IntVar()
        self.guiUtil.addComponent(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="Data are Orca files",
            grid_options={"row": 6, "column": 0, "sticky": "w", "padx": 10, "pady": (15, 0)},
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
        self.guiUtil.addComponent(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="Create images",
            grid_options={"row": 6, "column": 1, "sticky": "w", "padx": 10, "pady": (15, 0)},
            variable=self.create_images_val,
            font=self.root.font,
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2
        )

        self.process_files_button = self.guiUtil.addComponent(
            self,
            component_type="Button",
            frame=self.process_data_frame,
            text="Process files",
            grid_options={"row": 7, "column": 0, "sticky": "w", "padx": 10, "pady": (15, 0)},
            font=self.root.font,
            width=120,
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

        self.show_graphs_button = self.guiUtil.addComponent(
            self,
            component_type="Button",
            text="Show graph",
            frame=self.process_data_frame,
            grid_options={"row": 7, "column": 1, "sticky": "ew", "padx": (5, 10), "pady": (15, 0)},
            font=self.root.font,
            width=50,
            height=25,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="disabled",
            command=lambda: DataVisualiser(self.process_files.get_orbit_counts_df()).visualize()
        )

    def __createChooseLabelingMethods(self):
        self.guiUtil.addComponent(
            self,
            component_type="Label",
            frame=self.process_data_frame,
            text="Choose labeling methods (with label weight)",
            grid_options={"row": 9, "column": 0, "columnspan": 2, "sticky": "n"},
            font=self.root.font
        )

        self.hellinger_val = IntVar()
        self.hellinger_checkbox = self.guiUtil.addComponent(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="Hellinger",
            grid_options={"row": 10, "column": 0, "sticky": "w", "padx": (10, 0)},
            variable=self.hellinger_val,
            font=self.root.font,
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2,
            state="disabled",
            command=lambda: self.hellinger_weight.setDisabled(not self.hellinger_val.get())
        )

        self.hellinger_weight = self.guiUtil.addComponent(
            self,
            component_type="NumberInput",
            frame=self.process_data_frame,
            grid_options={"row": 10, "column": 1, "sticky": "e", "padx": (0, 10)},
        )
        self.hellinger_weight.setDisabled(True)

        self.netsimile_val = IntVar()
        self.netsimile_checkbox = self.guiUtil.addComponent(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="NetSimile (input data must be graphs)",
            grid_options={"row": 11, "column": 0, "sticky": "w", "padx": (10, 0), "pady": (0, 5)},
            variable=self.netsimile_val,
            font=self.root.font,
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2,
            state="disabled",
            command=lambda: self.netsimile_weight.setDisabled(not self.netsimile_val.get())
        )

        self.netsimile_weight = self.guiUtil.addComponent(
            self,
            component_type="NumberInput",
            frame=self.process_data_frame,
            grid_options={"row": 11, "column": 1, "sticky": "e", "padx": (0, 10), "pady": (0, 5)},
        )
        self.netsimile_weight.setDisabled(True)

        self.resnet_val = IntVar()
        self.resnet_checkbox = self.guiUtil.addComponent(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="Resnet (images has to be created)",
            grid_options={"row": 12, "column": 0, "columnspan": 2, "sticky": "w", "padx": 10, "pady": (0, 5)},
            variable=self.resnet_val,
            font=self.root.font,
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2,
            state="disabled",
            command=lambda: self.resnet_weight.setDisabled(not self.resnet_val.get())
        )

        self.resnet_weight = self.guiUtil.addComponent(
            self,
            component_type="NumberInput",
            frame=self.process_data_frame,
            grid_options={"row": 12, "column": 1, "sticky": "e", "padx": (0, 10), "pady": (0, 5)},
        )
        self.resnet_weight.setDisabled(True)

        self.kstest_val = IntVar()
        self.kstest_checkbox = self.guiUtil.addComponent(
            self,
            component_type="Checkbutton",
            frame=self.process_data_frame,
            text="Kolmogorov-Smirnov test",
            grid_options={"row": 13, "column": 0, "columnspan": 2, "sticky": "w", "padx": 10, "pady": (0, 5)},
            variable=self.kstest_val,
            font=self.root.font,
            checkbox_width=15,
            checkbox_height=15,
            corner_radius=7,
            fg_color=guiconst.COLOR_GREEN,
            hover_color=guiconst.COLOR_GREEN_HOVER,
            border_width=2,
            state="disabled",
            command=lambda: self.kstest_weight.setDisabled(not self.kstest_val.get())
        )

        self.kstest_weight = self.guiUtil.addComponent(
            self,
            component_type="NumberInput",
            frame=self.process_data_frame,
            grid_options={"row": 13, "column": 1, "sticky": "e", "padx": (0, 10)},
        )
        self.kstest_weight.setDisabled(True)

        self.count_similarities_button = self.guiUtil.addComponent(
            self,
            component_type="Button",
            text="Count similarities",
            frame=self.process_data_frame,
            grid_options={"row": 14, "column": 0, "columnspan": 2, "sticky": "ew", "padx": 10, "pady": (15, 0)},
            font=self.root.font,
            width=50,
            height=25,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="disabled",
            command=lambda: self.__handleComputeSimilarity()
        )

        self.label_similarity_button = self.guiUtil.addComponent(
            self,
            component_type="Button",
            text="Label similarities",
            frame=self.process_data_frame,
            grid_options={"row": 15, "column": 0, "columnspan": 2, "sticky": "ew", "padx": 10, "pady": (15, 10)},
            font=self.root.font,
            width=50,
            height=25,
            fg_color=guiconst.COLOR_GREY,
            hover_color=guiconst.COLOR_GREY_HOVER,
            state="disabled",
            command=lambda: self.__handleLabelSimilarities()
        )

        self.process_files_complete_label = self.guiUtil.addComponent(
            self,
            component_type="Label",
            frame=self.process_data_frame,
            text="",
            grid_options={"row": 16, "column": 0, "columnspan": 2, "sticky": "ew", "padx": 10, "pady": (15, 10)},
            font=self.root.font,
            anchor="center",
            text_color=guiconst.COLOR_GREEN
        )

    def run(self):
        self.__addHeader()
        self.__createProcessingDataFrame()
        self.guiUtil.createHorizontalLine(
            self.process_data_frame,
            width=self.process_data_frame_width - 20,
            row=8,
            column=0,
            columnspan=3,
            padx=5,
            pady=15,
            sticky="w"
        )
        self.__createChooseLabelingMethods()
        self.root.mainloop()
