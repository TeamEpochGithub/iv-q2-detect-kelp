"""Layout for the features page."""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from dash import html

from dashboard.utils import flatten_img, load_tiff, make_fig


def to_feature_df(img: npt.NDArray[np.float64 | np.float32 | np.int32], label: npt.NDArray[np.float64 | np.float32 | np.int32]) -> pd.DataFrame:
    """Convert an image and its label to a dataframe of features."""
    img_df = pd.DataFrame(flatten_img(img), columns=["SWIR", "NIR", "R", "G", "B", "Cloud", "Elevation"])
    img_df["Label"] = label.flatten()
    return img_df


def predictions_layout(image_id: str) -> html.Div:
    """Will display the predictions and the image channels for the given image ID."""
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
    latest_folder = Path("outputs/2024-02-14/10-34-51")
    preds_loc = latest_folder / "preds"
    if not os.path.exists(preds_loc / f"{image_id}_pred.tif"):
        return html.P(f"Predictions for {image_id} not found in {preds_loc}. This image is not in the test split")
    pred = load_tiff(preds_loc / f"{image_id}_pred.tif")
    pred = pred.reshape(350, 350)

    alpha = 0.5
    overlay_pred = rgb.copy()
    thresh = 0.5
    overlay_pred[pred > thresh, 0] = (1 - alpha) * overlay_pred[pred > thresh, 0] + alpha

    # read the csv in to a dataframe
    results_df = pd.read_csv(latest_folder / "results.csv")
    # get the dice_coef for this image
    dice_coef = results_df[results_df["image_key"] == image_id]["dice_coef"].to_numpy()[0]
    figs = [
        make_fig(pred, f"Prediction with dice_coef: {dice_coef:.2f}"),
        make_fig(overlay, "Kelp Overlay"),
        make_fig(overlay_pred, "Prediction Overlay"),
        make_fig(swir_nir_red, "SWIR/NIR/Red"),
    ]

    return html.Div(figs, style={"display": "flex"})
