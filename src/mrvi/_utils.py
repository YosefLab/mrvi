from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._types import MrVIReduction, _ComputeLocalStatisticsRequirements


def _parse_local_statistics_requirements(
    reductions: list[MrVIReduction],
) -> _ComputeLocalStatisticsRequirements:
    needs_mean_rep = False
    needs_sampled_rep = False
    needs_mean_dists = False
    needs_sampled_dists = False
    needs_normalized_dists = False

    ungrouped_reductions = []
    grouped_reductions = []

    for r in reductions:
        if r.input == "mean_representations":
            needs_mean_rep = True
        elif r.input == "sampled_representations":
            needs_sampled_rep = True
        elif r.input == "mean_distances":
            needs_mean_rep = True
            needs_mean_dists = True
        elif r.input == "sampled_distances":
            needs_sampled_rep = True
            needs_sampled_dists = True
        else:
            raise ValueError(f"Unknown reduction input: {r.input}")

        if r.group_by is None:
            ungrouped_reductions.append(r)
        else:
            grouped_reductions.append(r)

    return _ComputeLocalStatisticsRequirements(
        needs_mean_representations=needs_mean_rep,
        needs_sampled_representations=needs_sampled_rep,
        needs_mean_distances=needs_mean_dists,
        needs_sampled_distances=needs_sampled_dists,
        grouped_reductions=grouped_reductions,
        ungrouped_reductions=ungrouped_reductions,
    )


@jax.jit
def rowwise_max_excluding_diagonal(matrix):
    """Returns the rowwise maximum of a matrix excluding the diagonal."""
    assert matrix.ndim == 2
    num_cols = matrix.shape[1]
    mask = (1 - jnp.eye(num_cols)).astype(bool)
    return (jnp.where(mask, matrix, -jnp.inf)).max(axis=1)


def simple_reciprocal(w, eps=1e-6):
    """Convert distances to similarities via a reciprocal."""
    return 1.0 / (w + eps)


@partial(jax.jit, static_argnums=(2,))
def geary_c(
    w: jnp.ndarray,
    x: jnp.ndarray,
    similarity_fn: callable,
):
    """Computes Geary's C statistic from a distance matrix and a vector of values.

    Parameters
    ----------
    w
        distance matrix
    x
        vector of continuous values
    similarity_fn
        function that converts distances to similarities
    """
    # spatial weights are higher for closer points
    w_ = similarity_fn(w)
    w_ -= jnp.diag(jnp.diag(w_))
    num = x[:, None] - x[None, :]
    num = (num**2) * w_
    num = num / (2 * w_.sum())
    denom = x.var()
    return num.sum() / denom


@jax.jit
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


def compute_statistic(
    distances: np.ndarray | jnp.ndarray,
    node_colors: np.ndarray | jnp.ndarray,
    statistic: str = "geary",
    similarity_fn: callable = simple_reciprocal,
):
    """Computes a statistic for guided analyses.

    Parameters
    ----------
    distances
        square distance matrix between all observations
    node_colors
        observed covariate values for each observation
    statistic
        one of "geary" or "nn"
    similarity_fn
        function used to compute spatial weights for Geary's C statistic
    """
    distances_ = jnp.array(distances)
    node_colors_ = jnp.array(node_colors)

    stat_fn_kwargs = {}
    if statistic == "geary":
        stat_fn = geary_c
        stat_fn_kwargs["similarity_fn"] = similarity_fn
    elif statistic == "nn":
        stat_fn = nn_statistic
    assert stat_fn is not None

    t_obs = stat_fn(distances_, node_colors_, **stat_fn_kwargs)
    return t_obs


def permutation_test(
    distances: np.ndarray | jnp.ndarray,
    node_colors: np.ndarray | jnp.ndarray,
    statistic: str = "geary",
    similarity_fn: callable = simple_reciprocal,
    n_mc_samples: int = 1_000,
    selected_tail: str = "greater",
    random_seed: int = 0,
    use_vmap: bool = True,
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
    similarity_fn
        function used to compute spatial weights for Geary's C statistic
    n_mc_samples
        number of Monte Carlo samples for the permutation test
    selected_tail
        one of "less", "greater", to specify how to compute pvalues
    random_seed
        seed used to compute random sample permutations
    use_vmap
        whether or not to use vmap to compute pvalues
    """
    t_obs = compute_statistic(
        distances, node_colors, statistic=statistic, similarity_fn=similarity_fn
    )
    t_perm = []

    key = jax.random.PRNGKey(random_seed)
    keys = jax.random.split(key, n_mc_samples)

    @jax.jit
    def permute_compute(w, x, key):
        x_ = jax.random.permutation(key, x)
        return compute_statistic(w, x_, statistic=statistic, similarity_fn=similarity_fn)

    if use_vmap:
        t_perm = jax.vmap(permute_compute, in_axes=(None, None, 0), out_axes=0)(
            distances, node_colors, keys
        )
    else:
        permute_compute_bound = lambda key: permute_compute(distances, node_colors, key)
        t_perm = jax.lax.map(permute_compute_bound, keys)

    if selected_tail == "greater":
        cdt = t_obs > t_perm
    elif selected_tail == "less":
        cdt = t_obs < t_perm
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    pval = (1.0 + cdt.sum()) / (1.0 + n_mc_samples)
    return pval
