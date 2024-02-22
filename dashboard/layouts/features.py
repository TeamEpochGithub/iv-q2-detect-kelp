"""Layout for the features page."""

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

    # unstable
    # img_df["EVI"] = 2.5 * (img_df["NIR"] - img_df["R"]) / (img_df["NIR"] + 6 * img_df["R"] - 7.5 * img_df["B"] + 1)
    # # clip between -10 and 10
    # img_df["EVI"] = np.clip(img_df["EVI"], -10, 10)

    # not amazing
    # add more features here for kelp segmentation from landsat satellites
    # img_df["MDNI"] = (img_df["NIR"] - img_df["SWIR"]) / (img_df["NIR"] + img_df["SWIR"])

    # Floating algea index
    img_df["FAI1"] = img_df["NIR"] - 0.339*img_df["SWIR"] - 0.2*img_df["R"]

    # bad
    # nir - (red + (swir-red)*(nir-red)/(swir-red))
    # img_df["FAI2"] = img_df["NIR"] - (img_df["R"] + (img_df["SWIR"] - img_df["R"])*(img_df["NIR"] - img_df["R"])/(img_df["SWIR"] - img_df["R"]))

    img_df["AVI"] = (img_df["NIR"] - img_df["B"]) / (img_df["NIR"] + img_df["B"])

    # compute 2d fft of AVI, reshaping between pandas and numpy is a bit of a pain
    img_df["AVI_Fourier"] = np.abs(np.fft.fft2(img_df["AVI"].to_numpy().reshape(350, 350))).flatten()
    # clip up to 10
    img_df["AVI_Fourier"] = np.clip(img_df["AVI_Fourier"], 0, 10)

    # apply 2d fft, gaussian blur, then inverse 2d fft
    img_df["AVI_Fourier_smooth"] = np.abs(np.fft.ifft2(scipy.ndimage.gaussian_filter(np.fft.fft2(img_df["AVI"].to_numpy().reshape(350, 350)), 0))).flatten()

    # apply 2d fft abs twice
    img_df["AVI_Fourier2"] = np.abs(np.fft.fft2(np.abs(np.fft.fft2(img_df["AVI"].to_numpy().reshape(350, 350))))).flatten()

    # savi: ((nir - red) / (nir + red + 0.5)) * (1 + 0.5)
    L = 1 * 65535
    img_df["SAVI"] = ((img_df["NIR"] - img_df["R"]) / (img_df["NIR"] + img_df["R"] + L)) * (1 + L)

    # ndki (nir - green) / (nir + green)
    img_df["NDKI"] = (img_df["NIR"] - img_df["G"]) / (img_df["NIR"] + img_df["G"])

    # (nir - swir) / (nir + swir)
    img_df["NDNI"] = (img_df["NIR"] - 0.339*img_df["SWIR"])

    # Add the label to the dataframe
    img_df["Label"] = label.flatten()
    return img_df


def catboost_preds(img_df: pd.DataFrame, smoothing: float, thresh: float) -> tuple[npt.NDArray[np.float64], float]:
    """Use a catboost model to predict the kelp label from the features.

    :param img_df: dataframe of features and labels
    :param smoothing: smoothing factor for the predictions (gaussian std)
    :param thresh: threshold for the predictions
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

    label = img_df["Label"].to_numpy().reshape(350, 350)
    pred_bin = pred > thresh

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

    rgb = img_df[["R", "G", "B"]].to_numpy().reshape(350, 350, -1).astype(np.float32)
    rgb = rgb / 15000.0

    swir_nir_red = img_df[["SWIR", "NIR", "R"]].to_numpy().reshape(350, 350, -1).astype(np.float32)
    swir_nir_red = swir_nir_red / 22000.0

    label = img_df["Label"].to_numpy().reshape(350, 350)
    alpha = 0.5
    overlay = rgb.copy()
    overlay[label == 1, 0] = (1 - alpha) * overlay[label == 1, 0] + alpha

    thresh = 0.15
    pred, dice = catboost_preds(img_df, smoothing=0.6, thresh=thresh)
    pred = pred.reshape(350, 350)

    overlay_pred = rgb.copy()
    overlay_pred[pred > thresh, 0] = (1 - alpha) * overlay_pred[pred > thresh, 0] + alpha

    # Plot each image
    figs = [
        make_fig(pred, f"Catboost Predictions (Dice: {dice:.2f})"),
        make_fig(overlay, "Kelp Overlay"),
        make_fig(overlay_pred, "Pred overlay"),
        make_fig(swir_nir_red, "SWIR/NIR/Red"),
    ]
    figs_feats = [make_fig(img_df[feature].to_numpy().reshape(350, 350), feature) for feature in ["NDVI", "FAI1", "AVI","AVI_Fourier","AVI_Fourier_smooth","AVI_Fourier2"]]
    figs.extend(figs_feats)

    return html.Div(figs, style={"display": "flex"})
