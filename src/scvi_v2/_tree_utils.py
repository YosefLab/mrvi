from typing import Union

import numpy as np
import xarray as xr
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import squareform


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
