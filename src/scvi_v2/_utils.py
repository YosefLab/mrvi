from typing import Union

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def geary_c(
    w: jnp.ndarray,
    x: jnp.ndarray,
):
    """Computes Geary's C statistic from a distance matrix and a vector of values.

    Parameters
    ----------
    w
        distance matrix
    x
        vector of continuous values
    """
    # spatial weights are higher for closer points
    w_ = 1.0 / (w + 1e-6)
    num = x[:, None] - x[None, :]
    num = (num**2) * w_
    num = num / (2 * w_.sum())
    denom = x.var()
    return num.sum() / denom


# @jax.jit
def nn_statistic(
    w: jnp.ndarray,
    x: jnp.ndarray,
):
    """Computes differences of distances to nearest neighbor from the same group and from different groups.

    Parameters
    ----------
    w
        distance matrix
    x
        vector of discrete values
    """
    groups_mat = x[:, None] - x[None, :]
    groups_mat = (groups_mat == 0).astype(int)

    processed_mat1 = groups_mat - jnp.diag(groups_mat.diagonal())
    is_diff_group_or_id = processed_mat1 == 0
    # above matrix masks samples with different group or id, and diagonal elements
    penalties1 = jnp.zeros(w.shape)
    penalties1 += is_diff_group_or_id * 1e6
    d1 = w + penalties1
    distances_to_same = d1.min(axis=1)

    processed_mat2 = groups_mat
    is_same_group = processed_mat2 > 0
    # above matrix masks samples with same group (including the diagonal)
    penalties2 = jnp.zeros(w.shape)
    penalties2 += is_same_group * 1e6
    d2 = w + penalties2
    distances_to_diff = d2.min(axis=1)

    delta = distances_to_same - distances_to_diff
    return delta.mean()


def permutation_test(
    distances: Union[np.ndarray, jnp.ndarray],
    node_colors: Union[np.ndarray, jnp.ndarray],
    statistic: str = "geary",
    n_mc_samples: int = 1000,
    selected_tail: str = "greater",
    random_seed: int = 0,
):
    """Permutation test for guided analyses.

    Parameters
    ----------
    distances
        square distance matrix between all observations
    node_colors
        observed covariate values for each observation
    statistic
        one of "geary" or "nn"
    n_mc_samples
        number of Monte Carlo samples for the permutation test
    selected_tail
        one of "less", "greater", to specify how to compute pvalues
    random_seed
        seed used to compute random sample permutations
    """

    distances_ = jnp.array(distances)
    node_colors_ = jnp.array(node_colors)
    key = jax.random.PRNGKey(random_seed)

    if statistic == "geary":
        stat_fn = geary_c
    elif statistic == "nn":
        stat_fn = nn_statistic
    assert stat_fn is not None

    t_obs = stat_fn(distances, node_colors)
    t_perm = []

    keys = jax.random.split(key, n_mc_samples)

    @jax.jit
    def permute_compute(w, x, key):
        x_ = jax.random.permutation(key, x)
        return stat_fn(w, x_)

    t_perm = jax.vmap(permute_compute, in_axes=(None, None, 0), out_axes=0)(distances_, node_colors_, keys)

    if selected_tail == "greater":
        cdt = t_obs > t_perm
    elif selected_tail == "less":
        cdt = t_obs < t_perm
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    pval = (1.0 + cdt.sum()) / (1.0 + n_mc_samples)
    return pval
