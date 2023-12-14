import base64

import cv2
import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from dash_bootstrap_templates import load_figure_template

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
load_figure_template("Darkly")


# Replace this with your actual layout configuration
def create_layout():
    layout = html.Div([
        html.H1("Computer Vision Segmentation Dashboard"),

        # Image selection and display
        dcc.Dropdown(
            id='image-dropdown',
            options=[{'label': str(i), 'value': i} for i in range(1, 11)],  # Replace with your image IDs
            value=[1],
            multi=True
        ),

        # Image display blocks based on configuration
        html.Div(id='image-display')
    ])

    return layout


def display_fig(img: np.ndarray, title: str):
    fig = go.Figure()
    fig.add_trace(go.Image(z=img))
    fig.update_layout(title=title)

    # Adjusting margins
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    # Remove gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return dcc.Graph(figure=fig)

app.layout = create_layout()


# Callbacks to update the content dynamically
@app.callback(
    Output('image-display', 'children'),
    [Input('image-dropdown', 'value')]
)
def update_content(selected_images):

    # Replace this with your image display logic
    image_display_content = []
    for image_id in selected_images:
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = img / 2
        segmentation = img > 0.5

        # Plot each image with go imshow with title
        figs = [
            display_fig(img, "Original"),
            display_fig(processed, "Processed"),
            display_fig(segmentation, "Segmentation")
        ]

        # Arrange components in a div
        image_div = html.Div(figs, style={'display': 'flex'})
        # Add a title to the div
        full_div = html.Div([html.H3(f'Image {image_id}'), image_div])
        image_display_content.append(full_div)

    return image_display_content


if __name__ == '__main__':
    app.run_server(debug=True)
