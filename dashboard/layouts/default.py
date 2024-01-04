"""Default layout for the dashboard."""
from pathlib import Path

import numpy as np
from dash import html

from dashboard.utils import load_tiff, make_fig


def default_layout(image_id: str) -> html.Div:
    """Create the default layout for the dashboard, with RGB, IR, Clouds, Elevation, Kelp and Kelp overlay.

    :param image_id: ID of the image to display
    :return: the layout as html element
    """
    x_path = Path(f"./data/raw/train_satellite/{image_id}_satellite.tif")
    y_path = Path(f"./data/raw/train_kelp/{image_id}_kelp.tif")
    x = load_tiff(x_path)
    y = load_tiff(y_path)

    # put the channels in the last dimension
    x = np.moveaxis(x, 0, -1)
    y = np.squeeze(y)

    ir = x[:, :, (0, 1, 2)] / 22000
    rgb = x[:, :, (2, 3, 4)] / 15000

    # Create an overlay of the kelp on the RGB image
    alpha = 0.5
    overlay = rgb.copy()
    overlay[y == 1, 0] = (1 - alpha) * overlay[y == 1, 0] + alpha

    # Plot each image
    figs = [
        make_fig(overlay, "Kelp Overlay"),
        make_fig(ir, "SWIR/NIR/Red"),
        make_fig(rgb, "RGB"),
        make_fig(x[:, :, 5], "Clouds"),
        make_fig(x[:, :, 6], "Elevation"),
        make_fig(y, "Kelp"),
    ]

    return html.Div(figs, style={"display": "flex"})
