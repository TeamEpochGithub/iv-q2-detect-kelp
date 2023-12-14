from typing import Any
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
from polars import col


def create_production(train: pd.DataFrame) -> dbc.Row:
    """
    Create production graph using train data.
    :param train: The train data
    :return: The production graph
    """

    # Get groups for train
    grouped = train.groupby(['prediction_unit_id'])

    # Get list of groups
    groups = [{'label': str(name), 'value': name} for name in grouped.groups.keys()]

    production = dbc.Row([
        html.H1("Production"),
        dbc.Col([
            html.H3("Options"),
            html.Div([
                html.H4("Series"),
                dcc.Dropdown(id="prod-series-dropdown",
                             options=groups, value=groups[0]['label'], clearable=False)
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
        Output("production", "figure"),
        Input("prod-series-dropdown", "value")
    )
    def update_target(series_id: Any) -> Any:
        """ Create consumption graph using train data."""
        series = grouped.get_group(series_id).filter(col("is_consumption") == 0)
        fig = px.line(series, x="datetime", y="target")
        fig.update_layout(
            title="Production",
            xaxis_title="Date",
            yaxis_title="Production",
            legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        return fig

    return production
