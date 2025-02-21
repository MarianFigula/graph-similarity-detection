import plotly.graph_objects as go
from BusinessLogic.DataNormaliser.DataNormaliser import DataNormaliser


class DataVisualiser:
    def __init__(self, orbit_counts):
        self.orbit_counts = DataNormaliser(
            orbit_counts).log_scale_percentage_normalisation()
        self.categories = list(f"G{i}" for i in range(30))
        self.fig = None

    def create_scatter(self):
        self.fig = go.Figure()

        for col in self.orbit_counts:
            self.fig.add_trace(
                go.Scatter(
                    x=self.categories,
                    y=self.orbit_counts[col],
                    mode="lines+markers",
                    name=col,
                )
            )

        self.fig.update_layout(
            xaxis_title="Categories",
            yaxis_title="Values",
            showlegend=True,
        )
        self.fig.update_yaxes(type="log")

        self.fig.show()

    def visualize(self):
        self.create_scatter()
