import json
import math
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import plotly
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from matplotlib.gridspec import GridSpec
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from pyngrok import ngrok
from dash import Dash, html, dcc, callback, Output, Input
from consumption import create_consumption
from utils.colors import _get_colors

# Paths
RAW_DATA_PATH = Path("../data/raw")
PROC_DATA_PATH = Path("../data/processed")

# Metadata
UNIT_ID_COL = "prediction_unit_id"
TGT_PK_COLS = ["county", "is_business", "product_type"]
COORD_COL2ABBR = {"latitude": "lat", "longitude": "lon"}

# Load train data
train = pl.read_csv(RAW_DATA_PATH / "train.csv", try_parse_dates=True)
h_wth = pl.read_csv(
    RAW_DATA_PATH / "historical_weather.csv", try_parse_dates=True)
f_wth = pl.read_csv(RAW_DATA_PATH / "forecast_weather.csv",
                    try_parse_dates=True)


station_latlon_county_map = pl.read_csv(PROC_DATA_PATH / "county_lat_lon.csv")

CAST_COORD = [pl.col("lat").cast(pl.Float32), pl.col("lon").cast(pl.Float32)]
station_latlon_county_map = (
    station_latlon_county_map
    .drop("")
    .rename(COORD_COL2ABBR)
    .with_columns(*CAST_COORD)
)


def _cal_wth_local_mean(df_wth: pl.DataFrame, gp_keys: List[str]) -> pl.DataFrame:
    """Calculate county-level local weather stats.

    Only mean is derived now.

    Args:
        df_wth: weather data
        gp_keys: groupby keys

    Returns:
        df_wth_local_stats: county-level local weather stats
    """
    df_wth = df_wth.with_columns(*CAST_COORD)
    df_wth_local_stats = (
        df_wth
        .join(station_latlon_county_map, on=list(COORD_COL2ABBR.values()), how="left")
        .filter(pl.col("county").is_not_null())
        .group_by(gp_keys).mean()
        .select(*gp_keys, pl.exclude(gp_keys).name.suffix("_local_mean"))
    )

    return df_wth_local_stats


h_wth = h_wth.with_columns(pl.col("datetime") + pl.duration(days=1, hours=13))

h_wth = h_wth.rename(COORD_COL2ABBR)
gp_keys_h = ["datetime", "county"]
h_wth_local_stats = _cal_wth_local_mean(h_wth, gp_keys=gp_keys_h)

f_wth = f_wth.rename(COORD_COL2ABBR)
gp_keys_f = ["origin_datetime", "hours_ahead",
             "data_block_id", "forecast_datetime"] + ["county"]
f_wth_local_stats = _cal_wth_local_mean(f_wth, gp_keys_f)

f_wth_local_stats = (
    f_wth_local_stats
    .with_columns([
        pl.col("forecast_datetime")
        .dt.convert_time_zone("Europe/Bucharest").alias("datetime")
        .dt.replace_time_zone(None).cast(pl.Datetime("us"))
    ])
    .drop("forecast_datetime")
)

train = (
    train
    .join(h_wth_local_stats, on=gp_keys_h, how="left")
    .join(f_wth_local_stats, on=["data_block_id", "datetime", "county"], how="left", suffix="_f")
)


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

with open(PROC_DATA_PATH / "county_id2name.pkl", "rb") as f:
    county_id2name = pickle.load(f)
    county_name2id = {v: k for k, v in county_id2name.items()}
county_name2latlon = pl.read_parquet(
    PROC_DATA_PATH / "county_name2latlon.parquet")
train = train.with_columns(pl.col("county").cast(
    pl.Utf8).replace(county_id2name).alias("county_name"))

county_list = [f"{cid}-{name}" for cid, name in county_id2name.items()]
county_opts = [{"label": c, "value": c} for c in ["All"] + county_list]
county_opts_map = {c: [c] for c in county_list}
county_opts_map["All"] = county_list

# Colors for county central points
colors = _get_colors(16)
county_name2color = dict(zip((county_id2name.values()), colors))

fwth_feat_list = [
    "temperature",
    "dewpoint",
    "cloudcover_high",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_total",
    "10_metre_u_wind_component",
    "10_metre_v_wind_component",
    "direct_solar_radiation",
    "surface_solar_radiation_downwards",
    "snowfall",
    "total_precipitation"
]
fwth_feat_opts = [{"label": c, "value": c} for c in fwth_feat_list]
fwth_feat_opts_map = {c: [c] for c in fwth_feat_list}

numerical_columns = [
    var for var in train.to_pandas()._get_numeric_data().columns]

sidebar = html.Div(
    [
        dbc.Row(
            [
                html.H1('Settings')
            ],
            style={"height": "5vh"}, className='bg-primary text-white'
        ),
        dbc.Row(
            html.Div([
                html.H4("Select county:"),
                dcc.Dropdown(id="county-dropdown",
                             options=county_opts, value="All", clearable=False)
            ])
        ),
        dbc.Row(
            html.Div([
                html.H4("Select target type:"),
                dbc.RadioItems(
                    id="target-radio",
                    options=[{"label": x, "value": x}
                             for x in ["Consumption", "Production"]],
                    value="Consumption"
                )
            ])
        ),
        dbc.Row(
            html.Div([
                html.H4("Select forecast weather feature:"),
                dcc.Dropdown(id="fwth-feat-dropdown", options=fwth_feat_opts,
                             value=fwth_feat_list[0], clearable=False)
            ])
        ),
        dbc.Row([
            html.Div([
                html.H4("Select heatmap columns:"),
                dcc.Dropdown(id="heatmap-columns-dropdown", options=[{'label': x, 'value': x} for x in numerical_columns], multi=True,
                             value=numerical_columns, style={'width': '320px'})
            ])
        ])
    ],
    style={"height": "100vh", "padding": "2rem 1rem"}, className='bg-light'
)

content = html.Div(
    [
        dbc.Row([
            dcc.Graph(id="target"),
            dcc.Graph(id="wth")
        ]),
        dbc.Row([
            dcc.Graph(id="heatmap")
        ]),
        dbc.Row([
            dcc.Graph(id="map")
        ])
    ]
)

app.layout = html.Div([
    dbc.Row([
        html.H1("Exploratory Data Analysis", className="text-center")
    ]),
    create_consumption(train.to_pandas()),
    dbc.Row([
        dbc.Col(sidebar, width=3, className='bg-light'),
        dbc.Col(content, width=6),
    ]),
])

map_layout = go.Layout(
    autosize=True,
    hovermode="closest",
    mapbox_style="open-street-map",
    mapbox=dict(
        bearing=0,
        center=dict(lat=58.65, lon=24.95),
        pitch=0,
        zoom=6
    ),
    showlegend=False,
    margin=dict(l=5, r=20, t=20, b=5, pad=0)
)

heatmap = go.Heatmap(
    z=train[numerical_columns].to_pandas()._get_numeric_data().corr().values,
    x=train[numerical_columns].to_pandas()._get_numeric_data().corr().columns,
    y=train[numerical_columns].to_pandas()._get_numeric_data().corr().columns,
    colorscale='Viridis'
)


@callback(
    Output("target", "figure"),
    [Input("county-dropdown", "value"),
     Input("target-radio", "value")]
)
def update_target(slc_county, slc_target_type):
    """Update target sequences."""
    if slc_county == "All":
        slc_county_list = county_opts_map[county_list[0]]
    else:
        slc_county_list = county_opts_map[slc_county]
    slc_county_name = ["-".join(c.split("-")[1:]) for c in slc_county_list]
    is_cons = 1 if slc_target_type == "Consumption" else 0
    target = (
        train
        .filter((pl.col("county_name").is_in(slc_county_name)) & (pl.col("is_consumption") == is_cons))
        .sort("datetime")
    )
    fig = px.line(target, x="datetime", y="target", color="prediction_unit_id")
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


@callback(
    Output("wth", "figure"),
    [Input("county-dropdown", "value"),
     Input("fwth-feat-dropdown", "value")]
)
def update_weather(slc_county, slc_fwth_feat):
    if slc_county == "All":
        slc_county_list = county_opts_map[county_list[0]]
    else:
        slc_county_list = county_opts_map[slc_county]
    slc_county_name = ["-".join(c.split("-")[1:]) for c in slc_county_list]

    slc_fwth_feat_name = fwth_feat_opts_map[slc_fwth_feat][0]
    if f"{slc_fwth_feat_name}_local_mean_f" in train:
        feat_name_suffix = "local_mean_f"
    else:
        feat_name_suffix = "local_mean"
    slc_fwth_feat_name = f"{slc_fwth_feat_name}_{feat_name_suffix}"

    wth = (
        train
        .filter((pl.col("county_name").is_in(slc_county_name)))
        .select(["datetime", slc_fwth_feat_name])
        .unique()
        .sort("datetime")
    )
    fig = px.line(wth, x="datetime", y=slc_fwth_feat_name)
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig


@callback(
    Output("map", "figure"),
    Input("county-dropdown", "value")
)
def update_map(slc_county):
    slc_county_list = county_opts_map[slc_county]
    slc_county_name = ["-".join(c.split("-")[1:]) for c in slc_county_list]

    data = []
    # County boundary
    for r in county_name2latlon.drop_nulls().iter_rows(named=True):
        data.append(
            go.Scattermapbox(
                lat=r["lat_bound"],
                lon=r["lon_bound"],
                mode="lines",
                line=dict(width=1, color="gray"),
                name=r["county_name"],
                fill="toself" if r["county_name"] in slc_county_name else None,
            )
        )

    # County central point
    for county_name in sorted(slc_county_name):
        data.append(
            go.Scattermapbox(
                lat=county_name2latlon.filter(
                    pl.col("county_name") == county_name)["lat"],
                lon=county_name2latlon.filter(
                    pl.col("county_name") == county_name)["lon"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=county_name2color[county_name],
                    opacity=0.7
                ),
                name=county_name,
                hoverinfo="name"
            )
        )

    return {
        "data": data,
        "layout": map_layout
    }


@callback(
    Output("heatmap", "figure"),
    Input("heatmap-columns-dropdown", "value")
)
def update_heatmap(heatmap_columns):
    corr = train[heatmap_columns].to_pandas()._get_numeric_data().corr()

    heatmap = go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis'
    )

    return {"data": [heatmap], "layout": heatmap}


if __name__ == "__main__":
    app.run_server(debug=True)
