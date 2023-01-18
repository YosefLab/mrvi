import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from scvi_v2._tree_utils import TreeExplorer, compute_dendrogram_from_distance_matrix
from scvi_v2._utils import compute_statistic, permutation_test


def test_geary():
    # case with spatial correlation
    np.random.seed(0)
    pos = np.random.randn(100, 2)
    w = pairwise_distances(pos)
    x = pos[:, 0]

    assert compute_statistic(w, x, statistic="geary") < 1
    assert permutation_test(w, x, selected_tail="greater") < 0.05
    assert permutation_test(w, x, selected_tail="greater", use_vmap=False) < 0.05

    # case without expected
    ps = []
    for _ in range(10):
        x = np.random.randn(100)
        p = permutation_test(w, x, selected_tail="greater")
        ps.append(p)
        p_no_vmap = permutation_test(w, x, selected_tail="greater", use_vmap=False)
        ps.append(p_no_vmap)

    ps = np.array(ps)
    assert ps.max() >= 0.3


def test_nn():
    # case with spatial correlation
    np.random.seed(0)
    pos = np.random.randn(100, 2)
    w = pairwise_distances(pos)
    x = pos[:, 0] >= 0
    x = x.astype(int)
    assert compute_statistic(w, x, statistic="nn") < 0
    assert permutation_test(w, x, statistic="nn", selected_tail="greater") < 0.05
    assert permutation_test(w, x, statistic="nn", selected_tail="greater", use_vmap=False) < 0.05

    # case without expected
    ps = []
    for _ in range(10):
        x = np.random.randn(100) >= 0
        x = x.astype(int)
        p = permutation_test(w, x, statistic="nn", selected_tail="greater")
        ps.append(p)
        p_no_vmap = permutation_test(w, x, statistic="nn", selected_tail="greater", use_vmap=False)
        ps.append(p_no_vmap)
    ps = np.array(ps)
    assert ps.max() >= 0.3


def test_hierarchy():
    points = np.array([-2, -2.5, -2.1, 2.51, 2.5, 3]).reshape(-1, 1)
    Z = compute_dendrogram_from_distance_matrix(pairwise_distances(points))
    assert Z.shape == (5, 4)

    leaves_names = [f"leaf_{i}" for i in range(6)]
    tree_explorer = TreeExplorer(Z, leaves_labels=leaves_names)

    root_id = tree_explorer.root_id
    left_leaves = tree_explorer.get_left_leaves(root_id)
    right_leaves = tree_explorer.get_right_leaves(root_id)
    gt_left_leaves = {0, 1, 2}
    gt_right_leaves = {3, 4, 5}
    possible_cdt_1 = (set(left_leaves) == gt_left_leaves) and (set(right_leaves) == gt_right_leaves)
    possible_cdt_2 = (set(left_leaves) == gt_right_leaves) and (set(right_leaves) == gt_left_leaves)
    assert possible_cdt_1 or possible_cdt_2

    tree_explorer.compute_tree_coords()
    # tree_explorer.simple_plot()

    leaves_colors = pd.DataFrame(
        {
            "color1": ["case", "case", "case", "control", "control", "control"],
            "color2": np.random.randn(6),
        },
        index=leaves_names,
    )
    tree_explorer = TreeExplorer(
        Z,
        leaves_labels=leaves_names,
        leaves_metadata=leaves_colors,
    )
    tree_explorer.simple_plot()
