import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_bootstrap_templates import load_figure_template

from layouts import default_layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
load_figure_template("Darkly")

layouts = {
    'Default': default_layout
}


def load_tile_ids() -> list[str]:
    """Load the tile IDs from the metadata file, sorted"""
    df = pd.read_csv('./data/raw/metadata_fTq0l2T.csv')
    ids = df['tile_id'][df['in_train']==True].tolist()
    ids.sort()
    return ids


# Replace this with your actual layout configuration
def create_layout():
    ids = load_tile_ids()

    layout = html.Div([
        html.H1("Kelp Detection Dashboard"),

        # Description
        html.P("Select a configuration and one or more images to display."),

        # layout selection dropdown
        dcc.Dropdown(
            id='layout-dropdown',
            options=[{'label': k, 'value': k} for k in layouts.keys()],
            value='Default',
            style={'color': 'black'}
        ),

        # Image selection and display
        dcc.Dropdown(
            id='image-dropdown',
            options=[{'label': id, 'value': id} for id in ids],  # Replace with your image IDs
            value=[ids[0]],
            multi=True,
            style={'color': 'black'}
        ),

        # button to refresh with bootstrap styling
        dbc.Button("Refresh", id='refresh-button', color="primary", className="mr-1"),

        # Image display blocks based on configuration
        html.Div(id='image-display')
    ])

    return layout


app.layout = create_layout()


# Callbacks to update the content dynamically
@app.callback(
    Output('image-display', 'children'),
    [State('image-dropdown', 'value'),
     Input('layout-dropdown', 'value'),
     Input('refresh-button', 'n_clicks')]
)
def update_content(selected_images, selected_layout, n_clicks):
    # Replace this with your image display logic
    image_display_content = []
    for image_id in selected_images:
        image_div = layouts[selected_layout](image_id)

        # Add a centered title to the div
        full_div = html.Div([html.H3(f'Image {image_id}')] + [image_div])
        image_display_content.append(full_div)

    return image_display_content


if __name__ == '__main__':
    app.run_server(debug=True)
