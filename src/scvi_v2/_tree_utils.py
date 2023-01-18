from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import squareform


class TreeExplorer:
    """Utility class to explore the tree structure of a dendrogram computed with scipy.cluster.hierarchy.linkage."""

    def __init__(
        self,
        dendrogram: np.ndarray,
        leaves_labels: Optional[np.ndarray] = None,
    ):
        self.dendrogram = dendrogram
        self.n_points_total = dendrogram.shape[0] + 1
        self.leaves_labels = np.array(leaves_labels) if leaves_labels is not None else np.arange(self.n_points_total)
        self.node_labels = None
        self.init_node_labels()
        self._check_dendrogram_valid()

    def init_node_labels(self):
        """Initializes node labels."""
        assert self.leaves_labels.shape[0] == self.n_points_total
        self.node_labels = np.concatenate(
            [
                self.leaves_labels,
                np.array(
                    [
                        f"node_{i}"
                        for i in range(self.n_points_total, self.dendrogram.shape[0] + self.n_points_total + 1)
                    ]
                ),
            ]
        )

    def get_children(self, node_id):
        """Computes list of leaves under a node."""
        if node_id < self.n_points_total:
            return [node_id]
        im = int(node_id - self.n_points_total)
        left_children = self.dendrogram[im, 0]
        right_children = self.dendrogram[im, 1]
        return self.get_children(left_children) + self.get_children(right_children)

    def get_partial_leaves(self, node_id, which):
        """Computes list of leaves under the left or right child of a node."""
        assert which in ["left", "right"]
        idx = 0 if which == "left" else 1
        leaves_nodeids = self.get_children(self.dendrogram[node_id - self.n_points_total, idx])
        leaves_nodeids = self._check_indices_valid(leaves_nodeids)
        return leaves_nodeids

    def get_left_leaves(self, node_id):
        """Computes list of leaves under the left child of a node."""
        return self.get_partial_leaves(node_id, "left")

    def get_right_leaves(self, node_id):
        """Computes list of leaves under the right child of a node."""
        return self.get_partial_leaves(node_id, "right")

    def get_tree_splits(self, max_depth=5):
        """Computes left and right children of each node in the tree starting from the root."""
        children_pairs = {}
        # start id: 2n - 2

        current_node_id = self.root_id
        for _ in range(max_depth):
            dendrogram_id = current_node_id - self.n_points_total
            if dendrogram_id < 0:
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
        necessary_condition = self.dendrogram.shape[1] == 4
        if not necessary_condition:
            raise ValueError("Dendrogram must have 4 columns. This is not the case for the dendrogram provided.")

    @staticmethod
    def _check_indices_valid(indices):
        indices_int = np.array(indices)
        indices_int = indices_int.astype(int)
        if not np.all(indices_int == indices):
            raise ValueError("Indices must be integers.")
        return indices_int

    def node_name(self, nodeid):
        """Returns the name of a node."""
        return self.node_labels[nodeid]

    def compute_tree_coords(
        self,
    ):
        """Utils to compute easy-to-use coordinates for plotting a dendrogram.

        In particular, maps each node id to its (x,y) coordinates in the dendrogram plot.

        Taken from https://stackoverflow.com/questions/43513698/relation-between-dendrogram-plot-coordinates-and-clusternodes-in-scipy.

        Returns
        -------
        dict
            Dictionary containing the following keys:
            - nodeids_to_coords: maps each node id to its (x,y) coordinates in the dendrogram plot
            - dend_info: `scipy`'s dendrogram info, containing in particular the tree structure
            coordinates, in `icoord` and `dcoord`.
        """
        dend_info = hc.dendrogram(self.dendrogram, no_plot=True)

        def _flatten(my_list):
            return [item for sublist in my_list for item in sublist]

        # icoord and dcoord are (n, 4) arrays, where n is the number of nodes in the dendrogram
        # these four numbers characterize the position of the U-shape of the dendrogram
        icoord_flat = _flatten(dend_info["icoord"])
        dcoord_flat = _flatten(dend_info["dcoord"])
        leave_coords = [(x, y) for x, y in zip(icoord_flat, dcoord_flat) if y == 0]
        # leave ids are listed in ascending order according to their x-coordinate
        order = np.argsort([x for x, y in leave_coords])
        nodeids_to_coords = dict(zip(dend_info["leaves"], [leave_coords[idx] for idx in order]))

        children_to_parent_coords = {}
        for i, d in zip(dend_info["icoord"], dend_info["dcoord"]):
            icoord = (i[1] + i[2]) / 2
            dcoord = d[1]  # or d[2], should be the same this
            parent_coord = (icoord, dcoord)
            left_coord = (i[0], d[0])
            right_coord = (i[-1], d[-1])
            children_to_parent_coords[(left_coord, right_coord)] = parent_coord

        # traverse tree from leaves upwards and populate mapping ID -> (x,y)
        _, node_list = hc.to_tree(self.dendrogram, rd=True)
        missing_ids_in_mapper = range(len(dend_info["leaves"]), len(node_list))
        while len(missing_ids_in_mapper) > 0:
            for node_id in missing_ids_in_mapper:
                node = node_list[node_id]
                children_are_mapped = (node.left.id in nodeids_to_coords) and (node.right.id in nodeids_to_coords)
                if children_are_mapped:
                    left_coord = nodeids_to_coords[node.left.id]
                    right_coord = nodeids_to_coords[node.right.id]
                    nodeids_to_coords[node_id] = children_to_parent_coords[(left_coord, right_coord)]
            missing_ids_in_mapper = [node_id for node_id in range(len(node_list)) if node_id not in nodeids_to_coords]

        return {
            "nodeids_to_coords": nodeids_to_coords,
            "dend_info": dend_info,
        }

    def simple_plot(
        self,
        use_ids_as_labels=False,
    ):
        """Simple plot of the dendrogram."""
        coords = self.compute_tree_coords()
        fig, ax = plt.subplots()
        d_info = coords["dend_info"]
        node_coords = coords["nodeids_to_coords"]

        # provide annotations to the leaves
        if use_ids_as_labels:
            annotated_node_coords = node_coords
        else:
            annotated_node_coords = {}
            for node_id, node_coord in node_coords.items():
                new_node_name = self.node_name(node_id)
                annotated_node_coords[new_node_name] = node_coord

        # Plotting tree branches
        for i in range(len(d_info["icoord"])):
            plt.plot(d_info["icoord"][i], d_info["dcoord"][i], "k-")
        # Plotting nodes
        for node_id, (x, y) in annotated_node_coords.items():
            ax.plot(x, y, "ro")
            ax.annotate(str(node_id), (x, y), xytext=(0, -8), textcoords="offset points", va="top", ha="center")
        plt.axis("off")
        return fig, ax


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
