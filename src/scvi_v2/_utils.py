import jax
import jax.numpy as jnp


def geary_c(
    w: jnp.ndarray,
    x: jnp.ndarray,
):
    """Geary's C statistic.

    x: array of shape (n,)
    w: array of shape (n, n)
    """

    num = x[:, None] - x[None, :]
    num = (num**2) * w
    num = num / (2 * w.sum())
    denom = x.var()
    return num.sum() / denom


def nn_statistic(
    w: jnp.ndarray,
    x: jnp.ndarray,
):
    groups_mat = x[:, None] - x[None, :]
    groups_mat = (groups_mat == 0).astype(int)

    processed_mat1 = groups_mat - jnp.diag(groups_mat.diagonal())
    is_diff_group_or_id = processed_mat1 == 0
    penalties1 = jnp.zeros(w.shape)
    penalties1 += is_diff_group_or_id * jnp.inf
    d1 = w + penalties1
    distances_to_same = d1.min(axis=1)

    is_same_group = processed_mat1 > 0
    penalties2 = jnp.zeros(w.shape)
    penalties2 += is_same_group * jnp.inf
    d2 = w + penalties2
    distances_to_diff = d2.min(axis=1)

    delta = distances_to_same - distances_to_diff
    return delta.mean()


def permutation_test(distances, node_colors, statistic="geary", n_mc_samples=1000, alternative="less"):
    """Permutation test for guided analyses."""

    distances_ = jnp.array(distances)
    node_colors_ = jnp.array(node_colors)
    key = jax.random.PRNGKey(0)

    if statistic == "geary":
        stat_fn = geary_c
    elif statistic == "nn":
        stat_fn = nn_statistic
    assert stat_fn is not None

    t_obs = stat_fn(distances, node_colors)
    t_perm = []

    keys = jax.random.split(key, n_mc_samples)

    def permute_compute(w, x, key):
        x_ = jax.random.permutation(key, x)
        return stat_fn(w, x_)

    t_perm = jax.vmap(permute_compute, in_axes=(None, None, 0), out_axes=0)(distances_, node_colors_, keys)

    if alternative == "greater":
        cdt = t_obs > t_perm
    elif alternative == "less":
        cdt = t_obs < t_perm
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    pval = (1.0 + cdt.sum()) / (1.0 + n_mc_samples)
    return pval
