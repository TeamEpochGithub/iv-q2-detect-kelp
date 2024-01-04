"""Layout for the features page."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
from dash import html

from dashboard.utils import flatten_img, load_tiff, make_fig, unflatten_df


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


def catboost_preds(img_df: pd.DataFrame, smoothing: float) -> tuple[npt.NDArray[np.float64], float]:
    """Use a catboost model to predict the kelp label from the features.

    :param img_df: dataframe of features and labels
    :param smoothing: smoothing factor for the predictions (gaussian std)
    """
    from catboost import CatBoostClassifier

    cbm = CatBoostClassifier()
    cbm.load_model("./notebooks/catboost_model.cbm")

    # read the features txt to make sure they are in the same order
    with open("./notebooks/catboost_features.txt") as f:
        features = f.read().splitlines()

    pred = cbm.predict_proba(img_df[features])[:, 1]
    pred = pred.reshape(350, 350)

    # smooth with gaussian
    pred = scipy.ndimage.gaussian_filter(pred, smoothing)

    label = img_df["Label"].numpy().reshape(350, 350)
    pred_bin = pred > 0.2  # threshold found in notebook, changed manually with smoothing

    if (np.sum(pred_bin) + np.sum(label)) == 0:
        return pred, 1
    dice_coeff = 2 * np.sum(pred_bin * label) / (np.sum(pred_bin) + np.sum(label))
    return pred, dice_coeff


def features_layout(image_id: str) -> html.Div:
    """Show SWIR/NIR/RED, two features: land closeness and IR water normed, and a prediction from catboost.

    Land closeness is the inverse of the distance to land.
    There is a strange artefact, namely in empty images there always seems to be land in the top left corner.
    IR water normed is the SWIR/NIR/RED channels minus the median of the IR channels in the water of that image.
    The catboost predictions is a very naive test to see if these features are useful.
    It will train a per-pixel classification model on the first image it sees, and reuses the saved model for the rest.

    :param image_id: ID of the image to display
    :return: the layout as html element
    """
    x_path = Path(f"./data/raw/train_satellite/{image_id}_satellite.tif")
    y_path = Path(f"./data/raw/train_kelp/{image_id}_kelp.tif")
    x = load_tiff(x_path)
    y = load_tiff(y_path)

    img_df = to_feature_df(x, y)

    rgb = unflatten_df(img_df[["R", "G", "B"]]).astype(np.float32)
    rgb = rgb / 15000.0

    swir_nir_red = unflatten_df(img_df[["SWIR", "NIR", "R"]]).astype(np.float32)
    swir_nir_red = swir_nir_red / 22000.0

    label = unflatten_df(img_df["Label"]).squeeze()
    alpha = 0.5
    overlay = rgb.copy()
    overlay[label == 1, 0] = (1 - alpha) * overlay[label == 1, 0] + alpha

    pred, dice = catboost_preds(img_df, 0)
    pred = pred.reshape(350, 350)

    overlay_pred = rgb.copy()
    overlay_pred[pred > 0.2, 0] = (1 - alpha) * overlay_pred[pred > 0.2, 0] + alpha

    # Plot each image
    figs = [
        make_fig(pred, f"Catboost Predictions (Dice: {dice:.2f})"),
        make_fig(overlay, "Kelp Overlay"),
        make_fig(overlay_pred, "Pred overlay"),
        make_fig(swir_nir_red, "SWIR/NIR/Red"),
    ]
    figs_feats = [make_fig(unflatten_df(img_df[feature]).squeeze(), feature) for feature in ["ONIR", "LandCloseness"]]
    figs.extend(figs_feats)

    return html.Div(figs, style={"display": "flex"})
