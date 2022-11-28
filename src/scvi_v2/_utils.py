import jax
import jax.numpy as jnp


def geary_c(w, x):
    """Geary's C statistic.

    x: array of shape (n,)
    w: array of shape (n, n)
    """

    num = x[:, None] - x[None, :]
    num = (num**2) * w
    num = num / (2 * w.sum())
    denom = x.var()
    return num.sum() / denom


def permutation_test(distances, node_colors, statistic="geary", n_mc_samples=1000, alternative="less"):
    """Permutation test for guided analyses."""

    distances_ = jnp.array(distances)
    node_colors_ = jnp.array(node_colors)
    key = jax.random.PRNGKey(0)

    stat_fn = geary_c if statistic == "geary" else None
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
