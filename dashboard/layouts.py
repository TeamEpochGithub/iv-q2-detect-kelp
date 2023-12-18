import os.path
import warnings

import numpy as np
import numpy.typing as npt
import plotly.express as px
import plotly.graph_objects as go
import rasterio
from dash import dcc, html
from rasterio.errors import NotGeoreferencedWarning
import scipy


def load_tiff(path: str) -> npt.NDArray[np.int32]:
    """Load a tiff file as a numpy array
    :param path: path to the tiff file
    :return: numpy array of the tiff file"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(path) as f:
            img = f.read()
    return img


def make_fig(img: npt.NDArray[np.float32 | np.int32], title: str) -> dcc.Graph:
    """Take a numpy array of an image, with shape (X,Y,C) or (X,Y), values [0 to 1], and return a plotly figure.
    Is plotted as RGB or heatmap depending on the shape
    :param img: numpy array of the image
    :param title: title of the plot
    :return: plotly figure"""

    if len(img.shape) == 2:
        # if a single channel, show heatmap
        fig = px.imshow(img, color_continuous_scale='Inferno')
    else:
        # if multiple channels, plot as image
        fig = go.Figure()
        fig.add_trace(go.Image(z=img * 255))
    fig.update_layout(title=title, title_x=0.5, title_xanchor='center', width=450)

    # Adjusting margins
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    # Remove gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return dcc.Graph(figure=fig)


def default_layout(image_id: str) -> html.Div:
    """Create the default layout for the dashboard, with RGB, IR, Clouds, Elevation, Kelp and Kelp overlay.
    :param image_id: ID of the image to display
    :return: the layout as html element"""

    x_path = f'./data/raw/train_satellite/{image_id}_satellite.tif'
    y_path = f'./data/raw/train_kelp/{image_id}_kelp.tif'
    x = load_tiff(x_path)
    y = load_tiff(y_path)

    # put the channels in the last dimension
    x = np.moveaxis(x, 0, -1)
    y = np.squeeze(y)

    ir = x[:, :, (0, 1, 2)] / 65535
    rgb = x[:, :, (2, 3, 4)] / 65535

    # Create an overlay of the kelp on the RGB image
    alpha = 0.25
    overlay = rgb.copy()
    overlay[y == 1, 0] = (1 - alpha) * overlay[y == 1, 0] + alpha

    # Plot each image
    figs = [
        make_fig(ir, "SWIR/NIR/Red"),
        make_fig(rgb, "RGB"),
        make_fig(x[:, :, 5], "Clouds"),
        make_fig(x[:, :, 6], "Elevation"),
        make_fig(y, "Kelp"),
        make_fig(overlay, "Kelp Overlay")
    ]

    return html.Div(figs, style={'display': 'flex'})


def features_layout(image_id: str) -> html.Div:
    """Create the default layout for the dashboard, with RGB, IR, Clouds, Elevation, Kelp and Kelp overlay,
    and manual features.
    :param image_id: ID of the image to display
    :return: the layout as html element"""

    x_path = f'./data/raw/train_satellite/{image_id}_satellite.tif'
    y_path = f'./data/raw/train_kelp/{image_id}_kelp.tif'
    x = load_tiff(x_path)
    y = load_tiff(y_path)

    # put the channels in the last dimension
    x = np.moveaxis(x, 0, -1)
    y = np.squeeze(y)

    ir = x[:, :, (0, 1, 2)] / 65535
    rgb = x[:, :, (2, 3, 4)] / 65535

    # Create an overlay of the kelp on the RGB image
    alpha = 0.25
    overlay = rgb.copy()
    overlay[y == 1, 0] = (1 - alpha) * overlay[y == 1, 0] + alpha

    # compute watercolor, as median where elevation is below 1 and not nan
    water_mask = (x[:, :, 6] < 1) & (x[:, :, 0] >= 0)
    if np.sum(water_mask) == 0:
        water_color = np.zeros(3)
    else:
        all_colors = ir[water_mask].reshape(-1, 3)
        water_color = np.median(all_colors, axis=0)
    ir_water_normed = ir - water_color

    # distance to land with scipy distance transform
    land_mask = x[:, :, 6] > 0
    land_dist = scipy.ndimage.distance_transform_edt(~land_mask)
    land_closeness = 1 / (1 + land_dist*0.1)

    # rescale so that ir_water_normed is between 0 and 1 in the water
    normed_min = np.min(ir_water_normed[(land_dist > 5) & (x[:, :, 0] >= 0)])
    normed_max = np.max(ir_water_normed[(land_dist > 5) & (x[:, :, 0] >= 0)])
    ir_water_normed2 = (ir_water_normed - normed_min) / (normed_max - normed_min)

    # use catboost predictions as a feature, for simplicity, train it on the same image,
    # uses the three channels in ir_water_normed and land_closeness
    import catboost
    model = catboost.CatBoostClassifier()

    # flatten the image features and stack them
    X = np.stack([ir_water_normed[:, :, 0].flatten(),
                  ir_water_normed[:, :, 1].flatten(),
                  ir_water_normed[:, :, 2].flatten(),
                  land_closeness.flatten()], axis=-1)

    # load the model if it exists
    if os.path.exists('./data/processed/catboost_model.cbm'):
        model.load_model('./data/processed/catboost_model.cbm')
    else:
        y_ = y.flatten()
        model.fit(X, y_, verbose=True)

    # save the model
    model.save_model('./data/processed/catboost_model.cbm')

    # predict on the same image
    y_pred = model.predict_proba(X)[:, 1]
    y_pred = y_pred.reshape(y.shape)

    # apply a gaussian blur to the per-pixel predictions
    y_pred = scipy.ndimage.gaussian_filter(y_pred, sigma=1)

    # Compute dice coefficient
    y_pred_round = y_pred > 0.5
    intersection = np.sum(y_pred_round & y)
    union = np.sum(y_pred_round) + np.sum(y)
    if union == 0:
        dice = 1
    else:
        dice = 2 * intersection / union

    # Plot each image
    figs = [
        make_fig(ir, "SWIR/NIR/Red"),
        make_fig(land_closeness, "Land Closeness"),
        make_fig(overlay, "Kelp Overlay"),
        make_fig(ir_water_normed2, "ir_water_normed"),
        make_fig(y_pred, f"Catboost prediction (dice={dice:.2f})"),
    ]

    return html.Div(figs, style={'display': 'flex'})
