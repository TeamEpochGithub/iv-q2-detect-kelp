"""Layout for the features page."""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
from dash import html

from dashboard.utils import flatten_img, load_tiff, make_fig

def to_feature_df(img: npt.NDArray[np.float64 | np.float32 | np.int32], label: npt.NDArray[np.float64 | np.float32 | np.int32]) -> pd.DataFrame:
    """Convert an image and its label to a dataframe of features."""
    img_df = pd.DataFrame(flatten_img(img), columns=["SWIR", "NIR", "R", "G", "B", "Cloud", "Elevation"])
    img_df["Water"] = img_df["Elevation"] < 1

    img_df["NDWI"] = (img_df["G"] - img_df["NIR"]) / (img_df["G"] + img_df["NIR"])
    img_df["MNDWI"] = (img_df["G"] - img_df["SWIR"]) / (img_df["G"] + img_df["SWIR"])
    img_df["NDVI"] = (img_df["NIR"] - img_df["R"]) / (img_df["NIR"] + img_df["R"])

    # Land closeness, roughly inverse distance to land
    land_dist = scipy.ndimage.distance_transform_edt(img[6, :, :] <= 0)
    land_closeness = 1 / (1 + land_dist * 0.1)
    img_df["LandCloseness"] = land_closeness.flatten()

    # Offset features, using difference with median of all water pixels
    if img_df["Water"].sum() > 0:
        img_df["ONIR"] = img_df["NIR"] - img_df[img_df["Water"]]["NIR"].median()

        ROffset = img_df["R"] - img_df[img_df["Water"]]["R"].median()
        GOffset = img_df["G"] - img_df[img_df["Water"]]["G"].median()
        img_df["ODWI"] = GOffset - img_df["ONIR"]
        img_df["ODVI"] = img_df["ONIR"] - ROffset
    else:
        img_df["ONIR"] = 0
        img_df["ODWI"] = 0
        img_df["ODVI"] = 0

    # Add the label to the dataframe
    img_df["Label"] = label.flatten()
    return img_df


def predictions_layout(image_id: str) -> html.Div:

    """Will display the predictions and the image channels for the given image ID.
    It will also plot a mask of the predictions on top of the RGB channels."""
    x_path = Path(f"./data/raw/train_satellite/{image_id}_satellite.tif")
    y_path = Path(f"./data/raw/train_kelp/{image_id}_kelp.tif")
    x = load_tiff(x_path)
    y = load_tiff(y_path)

    img_df = to_feature_df(x, y)

    rgb = img_df[["R", "G", "B"]].to_numpy().reshape(350, 350, -1).astype(np.float32)
    rgb = rgb / 15000.0

    swir_nir_red = img_df[["SWIR", "NIR", "R"]].to_numpy().reshape(350, 350, -1).astype(np.float32)
    swir_nir_red = swir_nir_red / 22000.0

    label = img_df["Label"].to_numpy().reshape(350, 350)

    alpha = 0.5
    overlay = rgb.copy()
    overlay[label == 1, 0] = (1 - alpha) * overlay[label == 1, 0] + alpha

    # In outputs folder get the latest folder
    # In that folder again find the latest folder
    # Read the preds from that folder
    
    # get the folder names in the outputs folder
    output_dir = Path("./outputs")
    folders = [f for f in output_dir.iterdir() if f.is_dir()]
    # get the latest folder
    latest_folder = max(folders, key=os.path.getctime)

    # get the folder names in the latest folder
    folders = [f for f in latest_folder.iterdir() if f.is_dir()]
    # get the latest folder
    latest_folder = max(folders, key=os.path.getctime)

    # get the preds from the latest folder
    preds_loc = latest_folder / "preds"
    pred = load_tiff(preds_loc / f"{image_id}_pred.tif")
    pred = pred.reshape(350, 350)

    # make an overlay of preds on the rgb images
    overlay_pred = rgb.copy()
    overlay_pred[:, 0] = (1 - alpha) * overlay_pred[:, 0] + alpha

    figs = [
        make_fig(pred, f"Latest Predictions"),
        make_fig(overlay, "Kelp Overlay"),
        make_fig(overlay_pred, "Pred overlay"),
        make_fig(swir_nir_red, "SWIR/NIR/Red"),
    ]

    return html.Div(figs, style={"display": "flex"})
