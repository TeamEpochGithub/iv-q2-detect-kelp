"""Main dashboard file to run the dashboard."""

from collections.abc import Iterable

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_bootstrap_templates import load_figure_template

from dashboard.layouts.default import default_layout
from dashboard.layouts.features import features_layout
from dashboard.layouts.predictions import predictions_layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
load_figure_template("Darkly")


def load_tile_ids() -> list[str]:
    """Load the tile IDs from the metadata file, sorted.

    :return: list of tile IDs
    """
    metadata = pd.read_csv("./data/raw/metadata_fTq0l2T.csv")
    ids = metadata["tile_id"][metadata["in_train"] == 1].tolist()
    ids.sort()
    return ids


# Replace this with your actual layout configuration
def create_layout() -> html.Div:
    """Create the layout for the dashboard with input fields and image blocks depending on the configuration.

    :return: the layout as html element
    """
    layouts = {
        "Default": default_layout,
        "Features": features_layout,
        "Prediction": predictions_layout,
    }
    ids = load_tile_ids()

    return html.Div(
        [
            html.H1("Kelp Detection Dashboard"),
            # Description
            html.P("Select a configuration and one or more images to display."),
            # Layout selection dropdown
            dcc.Dropdown(
                id="layout-dropdown",
                options=[{"label": k, "value": k} for k in layouts],
                value="Default",
                style={"color": "black"},
            ),
            # Image selection and display
            dcc.Dropdown(
                id="image-dropdown",
                options=[{"label": image_id, "value": image_id} for image_id in ids],  # Replace with your image IDs
                value=[ids[0]],
                multi=True,
                style={"color": "black"},
            ),
            # button to refresh with bootstrap styling
            dbc.Button("Refresh", id="refresh-button", color="primary", className="mr-1"),
            # Image display blocks based on configuration
            html.Div(id="image-display"),
        ],
    )


# Callbacks to update the content dynamically
@app.callback(
    Output("image-display", "children"),
    [
        State("image-dropdown", "value"),
        Input("layout-dropdown", "value"),
        Input("refresh-button", "n_clicks"),
    ],
)
def update_content(selected_images: Iterable[str], selected_layout: str, n_clicks: int) -> list[html.Div]:  # noqa: ARG001
    """Update the content of the image display blocks based on the selected images and layout.

    :param selected_images: selected image IDs
    :param selected_layout: selected layout
    :param n_clicks: number of times the refresh button was clicked
    :return: list of html elements to display the images
    """
    # Replace this with your image display logic
    layouts = {
        "Default": default_layout,
        "Features": features_layout,
        "Prediction": predictions_layout,
    }
    image_display_content = []
    for image_id in selected_images:
        image_div = layouts[selected_layout](image_id)

        # Add a centered title to the div
        full_div = html.Div([html.H3(f"Image {image_id}"), image_div])
        image_display_content.append(full_div)

    return image_display_content
