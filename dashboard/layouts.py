import warnings

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
import rasterio
from dash import dcc, html
from rasterio.errors import NotGeoreferencedWarning


def load_tiff(path: str) -> npt.NDArray[np.int32]:
    """Load a tiff file as a numpy array
    :param path: path to the tiff file
    :return: numpy array of the tiff file"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(path) as f:
            img = f.read()
    return img


def make_fig(img: npt.NDArray[np.float32 | np.int32], title: str) -> dcc.Graph:
    """Take a numpy array of an image, with shape (X,Y,C) or (X,Y), values [0 to 1], and return a plotly figure.
    Is plotted as RGB or heatmap depending on the shape
    :param img: numpy array of the image
    :param title: title of the plot
    :return: plotly figure"""

    if len(img.shape) == 2:
        # if a single channel, show heatmap
        fig = px.imshow(img, color_continuous_scale='Inferno')
    else:
        # if multiple channels, plot as image
        fig = go.Figure()
        fig.add_trace(go.Image(z=img * 255))
    fig.update_layout(title=title, title_x=0.5, title_xanchor='center', width=450)

    # Adjusting margins
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    # Remove gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return dcc.Graph(figure=fig)


def default_layout(image_id: str) -> html.Div:
    """Create the default layout for the dashboard, with RGB, IR, Clouds, Elevation, Kelp and Kelp overlay.
    :param image_id: ID of the image to display
    :return: the layout as html element"""

    x_path = f'./data/raw/train_satellite/{image_id}_satellite.tif'
    y_path = f'./data/raw/train_kelp/{image_id}_kelp.tif'
    x = load_tiff(x_path)
    y = load_tiff(y_path)

    # put the channels in the last dimension
    x = np.moveaxis(x, 0, -1)
    y = np.squeeze(y)

    ir = x[:, :, (0, 1, 2)] / 65535
    rgb = x[:, :, (2, 3, 4)] / 65535

    # Create an overlay of the kelp on the RGB image
    alpha = 0.25
    overlay = rgb.copy()
    overlay[y == 1, 0] = (1 - alpha) * overlay[y == 1, 0] + alpha

    # Plot each image
    figs = [
        make_fig(ir, "SWIR/NIR/Red"),
        make_fig(rgb, "RGB"),
        make_fig(x[:, :, 5], "Clouds"),
        make_fig(x[:, :, 6], "Elevation"),
        make_fig(y, "Kelp"),
        make_fig(overlay, "Kelp Overlay")
    ]

    return html.Div(figs, style={'display': 'flex'})
