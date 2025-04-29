import os
import uuid

import plotly.graph_objects as go
import plotly.io as pio


class SnapShotOfGraphletsAsGraph:
    def __init__(self, orbit_counts_df, input_folder_path=None, output_folder_path=None):
        self.orbit_counts = orbit_counts_df
        self.categories = list(f"G{i}" for i in range(30))
        self.img_dir_id = str(uuid.uuid4())[:8]
        if not output_folder_path:
            output_folder_path = input_folder_path

        self.img_dir = f"{output_folder_path}/img_{self.img_dir_id}"
        self.fig = None

        os.makedirs(self.img_dir, exist_ok=True)

    def create_images(self):
        for col in self.orbit_counts:
            self.fig = go.Figure()

            self.fig.add_trace(
                go.Scatter(
                    x=self.categories,
                    y=self.orbit_counts[col],
                    mode="lines+markers",
                    name=col,
                )
            )
            self.fig.update_xaxes(showticklabels=False, title=None)
            self.fig.update_yaxes(showticklabels=False, title=None)

            self.fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                autosize=False,
            )
            self.fig.update_yaxes(type="log")

            img_path = os.path.join(self.img_dir, f"{col}.png")

            print("imgPath:", img_path)

            scale = 1.5
            pio.write_image(self.fig, img_path, format="png", scale=scale)

    def getImgDir(self):
        return self.img_dir
