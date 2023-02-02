import numpy as np
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
    def _check_pairs_same(pairs1, pairs2):
        left1, right1 = pairs1
        left2, right2 = pairs2
        possible_cdt_1 = (set(left1) == set(left2)) and (set(right1) == set(right2))
        possible_cdt_2 = (set(left1) == set(right2)) and (set(right1) == set(left2))
        return possible_cdt_1 or possible_cdt_2

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
    _check_pairs_same((left_leaves, right_leaves), (gt_left_leaves, gt_right_leaves))

    gt_pairs = [
        [[1, 2, 0], [3, 4, 5]],
        [[3, 4], [5]],
        [[1], [2, 0]],
    ]
    leaf_pairs = tree_explorer.get_tree_splits()
    for idx, gt_pair in enumerate(gt_pairs):
        _check_pairs_same(leaf_pairs[idx], gt_pair)
