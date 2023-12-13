import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px


def create_consumption(train: pd.DataFrame) -> dbc.Row:

    # Get groups for train
    grouped = train.groupby(['prediction_unit_id'])

    # Get list of groups
    groups = [{'label': str(name), 'value': name} for name in grouped.groups.keys()]
    
    consumption = dbc.Row([
        html.H1("Consumption"),
        dbc.Col([
            html.H3("Options"),
            html.Div([
                html.H4("Series"),
                dcc.Dropdown(id="series-dropdown",
                             options=groups, value=groups[0]['label'], clearable=False),
                html.H4("Sample"),
                dcc.Dropdown(id="con-sample", options=['M', 'D', 'H'], value='M', clearable=False)
            ])
        ],
            style={"padding": "2rem 1rem"},
            className='bg-light',
            width=3
        ),
        dbc.Col([
            html.Div([
                html.H4("Graph"),
                dcc.Graph(id="consumption")
            ])
        ], width=6)
    ])

    @callback(
        Output("consumption", "figure"),
        [Input("series-dropdown", "value"),
         Input("con-sample", "value")]
    )
    def update_target(series_id, sample):
        """ Create consumption graph using train data."""
        series = grouped.get_group(series_id)
        series = series.query("is_consumption == 1")
        series = series.set_index('datetime')
        series = series['target']
        series = series.resample(sample).mean()
        fig = px.line(series, y="target")
        fig.update_layout(
            title="Consumption",
            xaxis_title="Date",
            yaxis_title="Consumption",
            legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        return fig

    return consumption
