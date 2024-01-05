"""Common functions used in the dashboard."""

import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
import rasterio
from dash import dcc
from rasterio.errors import NotGeoreferencedWarning


def load_tiff(path: Path) -> npt.NDArray[np.int32]:
    """Load a tiff file as a numpy array.

    :param path: path to the tiff file
    :return: numpy array of the tiff file
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(path) as f:
            img = f.read()
    return img


def make_fig(img: npt.NDArray[np.float64 | np.float32 | np.int32], title: str) -> dcc.Graph:
    """Take a numpy array of an image, with shape (X,Y,C) or (X,Y), values [0 to 1], and return a plotly figure.

    It is plotted as RGB or heatmap depending on the shape.

    :param img: numpy array of the image
    :param title: title of the plot
    :return: plotly figure
    """
    if len(img.shape) == 2:  # If a single channel, show heatmap
        fig = px.imshow(img, color_continuous_scale="Inferno")
        fig.update_layout(coloraxis_showscale=False)
    else:  # If multiple channels, plot as image
        fig = go.Figure()
        fig.add_trace(go.Image(z=np.power(img, 2.2) * 255))
    fig.update_layout(title=title, title_x=0.5, title_xanchor="center", width=450)

    # Adjusting margins
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0})

    # Remove gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return dcc.Graph(figure=fig)


def flatten_img(img: npt.NDArray[np.float64 | np.float32 | np.int32]) -> npt.NDArray[np.float64 | np.float32 | np.int32]:
    """Flatten an image to have the shape (N, C).

    :param img: numpy array of the image
    """
    img = img.transpose(1, 2, 0)
    return img.reshape(-1, img.shape[2])
