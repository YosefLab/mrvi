from typing import Union

import numpy as np
import xarray as xr
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import squareform


class TreeExplorer:
    """Utility class to explore the tree structure of a dendogram computed with scipy.cluster.hierarchy.linkage."""

    def __init__(
        self,
        dendogram,
    ):
        self.dendogram = dendogram
        self.n_points_total = dendogram.shape[0] + 1
        self._check_dendrogram_valid()

    def get_children(self, node_id):
        """Computes list of leaves under a node."""
        if node_id < self.n_points_total:
            return [node_id]
        im = int(node_id - self.n_points_total)
        left_children = self.dendogram[im, 0]
        right_children = self.dendogram[im, 1]
        return self.get_children(left_children) + self.get_children(right_children)

    def get_left_leaves(self, node_id):
        """Computes list of leaves under the left child of a node."""
        return self.get_children(self.dendogram[node_id - self.n_points_total, 0])

    def get_right_leaves(self, node_id):
        """Computes list of leaves under the right child of a node."""
        return self.get_children(self.dendogram[node_id - self.n_points_total, 1])

    def get_tree_splits(self, max_depth=5):
        """Computes left and right children of each node in the tree starting from the root."""
        children_pairs = {}
        # start id: 2n - 2

        current_node_id = self.root_id
        for _ in range(max_depth):
            dendogram_id = current_node_id - self.n_points_total
            if dendogram_id < 0:
                break
            children_pairs[current_node_id] = (
                self.get_left_leaves(current_node_id),
                self.get_right_leaves(current_node_id),
            )
            current_node_id -= 1
        return children_pairs

    @property
    def root_id(self):
        """Returns the id of the root node."""
        return self.n_points_total * 2 - 2

    def _check_dendrogram_valid(self):
        necessary_condition = self.dendogram.shape[1] == 4
        if not necessary_condition:
            raise ValueError("Dendogram must have 4 columns. This is not the case for the dendogram provided.")


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
    dendogram = hc.linkage(dists_1d, method=linkage_method)
    dendogram = hc.optimal_leaf_ordering(dendogram, dists_1d)
    return dendogram
