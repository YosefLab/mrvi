from typing import Union

import ete3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.colors import rgb2hex
from scipy.cluster import hierarchy as hc
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import squareform


def linkage_to_ete(linkage_obj):
    """Converts to ete3 tree representation."""
    R = to_tree(linkage_obj)
    root = ete3.Tree()
    root.dist = 0
    root.name = "root"
    item2node = {R.get_id(): root}
    to_visit = [R]

    while to_visit:
        node = to_visit.pop()
        cl_dist = node.dist / 2.0

        for ch_node in [node.get_left(), node.get_right()]:
            if ch_node:
                ch_node_id = ch_node.get_id()
                ch_node_name = f"t{int(ch_node_id) + 1}" if ch_node.is_leaf() else str(ch_node_id)
                ch = ete3.Tree()
                ch.dist = cl_dist
                ch.name = ch_node_name

                item2node[node.get_id()].add_child(ch)
                item2node[ch_node_id] = ch
                to_visit.append(ch_node)
    return root


def convert_pandas_to_colors(metadata: pd.DataFrame):
    """Converts a pandas dataframe to hex colors."""

    def _get_colors_from_categorical(x):
        return np.array([rgb2hex(plt.cm.tab10(i)) for i in x])

    def _get_colors_from_continuous(x):
        return np.array([rgb2hex(plt.cm.viridis(i)) for i in x])

    dtypes = metadata.dtypes
    colors_mapper = {}
    for col in metadata.columns:
        if dtypes[col] == "object":
            cats = metadata[col].astype("category").cat.codes
            colors = _get_colors_from_categorical(cats)
        elif dtypes[col] == "category":
            colors = _get_colors_from_categorical(metadata[col])
        else:
            scales = (metadata[col] - metadata[col].min()) / (metadata[col].max() - metadata[col].min())
            colors = _get_colors_from_continuous(scales)
        colors_mapper[col] = colors
    return pd.DataFrame(colors_mapper, index=metadata.index)


def compute_dendrogram_from_distance_matrix(
    distance_matrix: Union[np.ndarray, xr.DataArray],
    linkage_method: str = "complete",
    symmetrize: bool = True,
    convert_to_ete: bool = False,
):
    """Computes a dendrogram from a distance matrix.

    Parameters
    ----------
    distance_matrix :
        distance matrix
    linkage_method :
        linkage method for hierarchical clustering
    """
    distance_matrix_ = distance_matrix.copy()
    if symmetrize:
        assert np.allclose(distance_matrix_, distance_matrix_.T)
        distance_matrix_ = (distance_matrix_ + distance_matrix_.T) / 2
    dists_1d = squareform(distance_matrix_, checks=False)
    dendrogram = hc.linkage(dists_1d, method=linkage_method)
    dendrogram = hc.optimal_leaf_ordering(dendrogram, dists_1d)
    if convert_to_ete:
        dendrogram = linkage_to_ete(dendrogram)
    return dendrogram


def plot_distance_matrix(
    distance_matrix: xr.DataArray,
    linkage_method: str = "complete",
    metadata: pd.DataFrame = None,
    colors: pd.DataFrame = None,
):
    """Plots a distance matrix.

    Parameters
    ----------
    distance_matrix :
        2d distance matrix
    linkage_method :
        linkage method for hierarchical clustering
    metadata :
        DataFrame containing categorical or continuous values.
        The dataframe must have the same index as the distance matrix coordinates.
    colors :
        DataFrame containing colors in hex format.
        The dataframe must have the same index as the distance matrix coordinates.
    """
    if metadata is not None:
        assert colors is None
        colors = convert_pandas_to_colors(metadata)
    assert distance_matrix.ndim == 2
    dendrogram = compute_dendrogram_from_distance_matrix(
        distance_matrix,
        linkage_method=linkage_method,
        symmetrize=True,
    )

    fig = sns.clustermap(distance_matrix.to_pandas(), row_linkage=dendrogram, col_linkage=dendrogram, row_colors=colors)
    return fig
