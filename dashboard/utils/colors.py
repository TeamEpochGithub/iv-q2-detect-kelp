import plotly
import plotly.graph_objects as go


def _get_colors(n_colors: int = 16) -> list[str]:
    """Reture color list for visualizing categorical column.
    :param n_colors: The number of colors to return
    :return: The list of colors
    """
    colors = plotly.colors.sample_colorscale(
        "Bluered", [i / 15 for i in range(16)])

    # Visualize color palette
    fig = go.Figure()
    for i, c in enumerate(colors):
        fig.add_bar(x=[i], y=[1], marker_color=c, showlegend=False, name=c)
    fig.update_layout(
        title={"text": f"Color Palette of {n_colors} Colors", "yanchor": "top"},
        height=50,
        margin=dict(l=0, r=0, t=30, b=0),
        bargap=0
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return colors
