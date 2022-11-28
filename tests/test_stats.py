import numpy as np
from scvi_v2._utils import permutation_test
from sklearn.metrics import pairwise_distances


def test_geary():
    # case with spatial correlation
    pos = np.random.randn(100, 2)
    w = pairwise_distances(pos)
    x = pos[:, 0]
    assert permutation_test(w, x) < 0.05

    # case without expected
    ps = []
    for _ in range(10):
        x = np.random.randn(100)
        p = permutation_test(w, x)
        ps.append(p)
    ps = np.array(ps)
    assert ps.max() >= 0.3


def test_nn():
    # case with spatial correlation
    pos = np.random.randn(100, 2)
    w = pairwise_distances(pos)
    x = pos[:, 0] >= 0
    x = x.astype(int)
    assert permutation_test(w, x, statistic="nn", alternative="greater") < 0.05

    # case without expected
    ps = []
    for _ in range(10):
        x = np.random.randn(100) >= 0
        x = x.astype(int)
        p = permutation_test(w, x, statistic="nn", alternative="greater")
        ps.append(p)
    ps = np.array(ps)
    assert ps.max() >= 0.3
