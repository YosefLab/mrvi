import numpy as np
from scvi_v2._utils import compute_statistic, permutation_test
from sklearn.metrics import pairwise_distances


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
    assert (
        permutation_test(w, x, statistic="nn", selected_tail="greater", use_vmap=False)
        < 0.05
    )

    # case without expected
    ps = []
    for _ in range(10):
        x = np.random.randn(100) >= 0
        x = x.astype(int)
        p = permutation_test(w, x, statistic="nn", selected_tail="greater")
        ps.append(p)
        p_no_vmap = permutation_test(
            w, x, statistic="nn", selected_tail="greater", use_vmap=False
        )
        ps.append(p_no_vmap)
    ps = np.array(ps)
    assert ps.max() >= 0.3
