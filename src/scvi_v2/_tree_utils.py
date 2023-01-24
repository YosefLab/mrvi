from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import rgb2hex
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import squareform


def convert_pandas_to_colors(metadata: pd.DataFrame):
    """Initializes leaves colors."""

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
            colors = _get_colors_from_continuous(metadata[col])
        else:
            scales = (metadata[col] - metadata[col].min()) / (metadata[col].max() - metadata[col].min())
            colors = _get_colors_from_continuous(scales)
        colors_mapper[col] = colors
    return pd.DataFrame(colors_mapper, index=metadata.index)


def compute_dendrogram_from_distance_matrix(
    distance_matrix: Union[np.ndarray, xr.DataArray],
    linkage_method: str = "complete",
    symmetrize: bool = True,
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
        distance_matrix_ = (distance_matrix_ + distance_matrix_.T) / 2
    dists_1d = squareform(distance_matrix_)
    dendrogram = hc.linkage(dists_1d, method=linkage_method)
    dendrogram = hc.optimal_leaf_ordering(dendrogram, dists_1d)
    return dendrogram
