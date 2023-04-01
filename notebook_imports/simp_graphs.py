import typing
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field


class CSS_COLORS(Enum):
    red = 'red'
    blue = 'blue'
    green = 'green'
    white = 'white'
    black = 'black'


class BarSet(BaseModel):
    label: str
    values: pd.Series
    # tooltips: TooltipSeriesInfo | None = None
    # color_override: CSSHex | settings.CSS_COLORS | None = None

    class Config:
        arbitrary_types_allowed = True


class BarGraph(BaseModel):
    sets: list[BarSet]
    labels: list[str]
    title: str = ""
    orientation: str = "vertical"
    value_axis_label: str = ""
    label_axis_label: str = ""
    bar_limit: int = 300

    class Config:
        arbitrary_types_allowed = True


class AxisThreshold(BaseModel):
    x_vals: pd.Series
    y_vals: pd.Series
    curve_type: str = "curved"
    # above_color_override: CSSHex | settings.CSS_COLORS | None = None
    # below_color_override: CSSHex | settings.CSS_COLORS | None = None
    # data_point_limit: int = DEFAULT_MAX_DATA_ROWS
    label: str = "Threshold"
    no_tooltip: bool = True

    class Config:
        arbitrary_types_allowed = True

class AxisTrace(BaseModel):
    x: pd.Series | np.ndarray | typing.Literal["minmax"]
    y: pd.Series | np.ndarray | typing.Literal["minmax"]
    x_axis_id: str = "default"
    y_axis_id: str = "default"
    thresholds: list[AxisThreshold] = Field(default_factory=list)

    label: str = ""
    trace_style: str = "scatter"
    line_type: str = "curved"
    line_width: float = 3
    point_size: float = 3
    point_shape: str = "circle"
    color_override: CSS_COLORS | None = None
    # hide_points: bool = False
    # fill_area: bool = False
    # hoverable: bool = True
    # data_point_limit: int = DEFAULT_MAX_DATA_ROWS
    # gradient_vals: pd.Series | None = None
    # gradient_in_tooltip: bool = True
    # tooltips: TooltipSeriesInfo | None = None

    class Config:
        arbitrary_types_allowed = True

class Axis(BaseModel):
    label: str = ""
    good_default_show: bool = True

    # data_type: AxisDataType_T = "raw"
    # utc_axis_format_getter: typing.Callable[
    #     [dm.timezone.timedelta], str
    # ] = default_utc_axis_format_getter
    # axis_range_getter: typing.Callable[
    #     [float, float, AxisDataType_T], tuple[float, float]
    # ] = default_axis_range_getter

class AxisGraph(BaseModel):
    traces: list[AxisTrace]
    title: str = ""
    x_axes: dict[str, Axis] = Field(default_factory=dict)
    y_axes: dict[str, Axis] = Field(default_factory=dict)
    x_label: str = ""
    y_label: str = ""
    smoothing_config: int | None = None

    color_gradient_axis_title: str = ""
    # color_gradient_type: AxisGradientType_T = "raw"
    # color_gradient_range_colors: list[CSSHex | settings.CSS_COLORS] | None = None
    default_normalise_subplots: bool = False
    show_voronoi: bool = False

    class Config:
        arbitrary_types_allowed = True


class SimpGraph(BaseModel):
    unbuilt_graphs: dict[str, AxisGraph | BarGraph] = {}

    def add(self, graph: AxisGraph | BarGraph, name: str):
        assert name not in self.unbuilt_graphs, "Graph name already exists"
        assert isinstance(graph, AxisGraph) or isinstance(
            graph, BarGraph
        ), "Can only add AxisGraph or BarGraph with Matplotlib on Simp Graphs"
        self.unbuilt_graphs[name] = graph

    def build_notebook(self, fig_size=(20, 10), padding=5):
        # TODO color override
        # TODO supports multiple bar sets
        if len(self.unbuilt_graphs.keys()) < 1:
            print("No unbuilt graphs to convert into matplot")
            return False
        unbuilt_graphs = list(self.unbuilt_graphs.values())
        unbuilt_labels = list(self.unbuilt_graphs.keys())
        n_subs = len(unbuilt_graphs)
        if n_subs == 0:
            print("No graphs to plot")
            return
        idxs = [1, 2, 4, 6, 9, 12, 16, 20, 25]
        sub_shape = [(1, 1), (1, 2), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (5, 5)]
        ref = 0
        for i, idx in enumerate(idxs):
            if n_subs == idx:
                ref = i
            elif n_subs > idx:
                ref = i + 1
        shape = sub_shape[ref]
        cols = shape[1]
        rows = shape[0]
        fig, ax = plt.subplots(rows, cols, figsize=fig_size)
        fig.tight_layout(pad=padding)
        i = 0

        for col in range(cols):
            for row in range(rows):
                if i >= len(unbuilt_graphs):
                    break
                graph = unbuilt_graphs[i]
                traces = graph.traces if isinstance(graph, AxisGraph) else graph.sets
                # bar_gap, bar_width = 0.5, 0.5
                assert type(traces) == list
                for trace in list(traces):
                    if rows == 1 and cols == 1:
                        element = ax
                    elif rows == 1:
                        element = ax[col]
                    else:
                        element = ax[row][col]

                    if isinstance(graph, BarGraph):
                        assert type(graph.labels[0]) == str, "Bar graph labels must be categorical"

                        if len(traces) > 1:
                            print("currently only supports one bar set in matplot")

                        element.bar(
                            pd.Series(graph.labels),
                            trace.values,
                            label=trace.label,
                            width=0.5,
                            bottom=None,
                            align="center"
                            # color=trace.color
                        )
                        x_label = graph.label_axis_label
                        y_label = graph.value_axis_label
                    else:
                        if trace.trace_style == "line":
                            element.plot(trace.x, trace.y, label=trace.label, color=trace.color_override.value)  # color=trace.color

                        if trace.trace_style == "scatter":
                            element.scatter(
                                trace.x, trace.y, label=trace.label, color=trace.color_override.value
                            )  # color=trace.color

                        x_label = graph.x_label
                        y_label = graph.y_label

                    if not element.get_xlabel():
                        element.set_xlabel(x_label)
                    if not element.get_ylabel():
                        element.set_ylabel(y_label)
                    if not element.get_title():
                        tit = graph.title
                        if tit == "":
                            tit = unbuilt_labels[i]
                        element.set_title(tit)
                    # element["showlegend"] = True
                    element.legend(loc="upper left")
                i += 1
        return fig.show()