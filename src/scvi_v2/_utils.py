from functools import partial
from typing import Callable, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from pydeseq2 import DeseqDataSet

from ._types import MrVIReduction, _ComputeLocalStatisticsRequirements


def _parse_local_statistics_requirements(
    reductions: List[MrVIReduction],
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
        elif r.input == "normalized_distances":
            needs_sampled_rep = True
            needs_normalized_dists = True
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
        needs_normalized_distances=needs_normalized_dists,
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
    similarity_fn: Callable,
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
    distances: Union[np.ndarray, jnp.ndarray],
    node_colors: Union[np.ndarray, jnp.ndarray],
    statistic: str = "geary",
    similarity_fn: Callable = simple_reciprocal,
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
    distances: Union[np.ndarray, jnp.ndarray],
    node_colors: Union[np.ndarray, jnp.ndarray],
    statistic: str = "geary",
    similarity_fn: Callable = simple_reciprocal,
    n_mc_samples: int = 1000,
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
    t_obs = compute_statistic(distances, node_colors, statistic=statistic, similarity_fn=similarity_fn)
    t_perm = []

    key = jax.random.PRNGKey(random_seed)
    keys = jax.random.split(key, n_mc_samples)

    @jax.jit
    def permute_compute(w, x, key):
        x_ = jax.random.permutation(key, x)
        return compute_statistic(w, x_, statistic=statistic, similarity_fn=similarity_fn)

    if use_vmap:
        t_perm = jax.vmap(permute_compute, in_axes=(None, None, 0), out_axes=0)(distances, node_colors, keys)
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


def extract_gene_and_cell_clusters(
    myadata: sc.AnnData,
    mylfcs: xr.DataArray,
    extend_nhops: int=2,
    score_threshold: float=0.5,
    score_quantile: float=None,
    u_repkey: str="X_u",
    plot: bool=True,
    neighbors_kwargs: dict=None,
):
    """
    Extract gene and cell clusters from the LFCs.

    This function clusters genes based on a cell by gene matrix of LFCs, to identify
    modules of interacting genes.
    It then identifies cell sets that have upregulated genes for each gene cluster.

    This function builds three quantities:
    1. gene clusters: genes that are close to each other in the LFC space, stored in
    `myadata.var['gene_clusters']`
    2. cell sets: Sets of cells that have upregulated genes for a given gene cluster,
    3. extended cell sets: Sets of cells, and their immediate neighbors in the u space, that have upregulated genes for a given gene cluster.

    Parameters
    ----------
    myadata
        AnnData object.
    mylfcs
        LFCs, in a DataArray format, corresponding to a cell by gene matrix.
    extend_nhops
        Number of hops to extend the gene clusters, used to construct extended cell sets.
        In particular, starting from an initial cell set, we add all k-nearest neighbors `extend_nhops` times.
    score_threshold
        Threshold for the cell score to construct cell sets.
        It corresponds to the percentage of genes in the gene cluster that are upregulated.
    score_quantile
        Quantile of the cell score to construct cell sets, based on quantiles of the cell score distribution instead of a fixed threshold.
        Replaces `score_threshold` if specified.
    u_repkey
        Key in `myadata.obsm` that contains the u embeddings. Used to construct extended cell sets.
    plot
        Whether to plot the results.
    """

    lfc_ad = sc.AnnData(
        X=mylfcs.values,
        obs=myadata.obs,
        var=myadata.var,
    )

    # estimate gene clusters
    gene_include_rule = np.logical_and(myadata.X.mean(0) > 0.01, lfc_ad.X.std(0) >= 0.1)
    lfc_ad_transpose = lfc_ad[:, gene_include_rule].copy().T

    if neighbors_kwargs is None:
        neighbors_kwargs = {
            "metric": "cosine",
            "n_neighbors": 15,
        }
    sc.pp.neighbors(lfc_ad_transpose, use_rep="X", **neighbors_kwargs)
    sc.tl.leiden(lfc_ad_transpose, resolution=0.4, key_added="gene_leiden")
    if plot:
        sc.tl.umap(lfc_ad_transpose, min_dist=0.1)
        sc.pl.umap(lfc_ad_transpose, color="gene_leiden")

    # assign gene clusters
    myadata.var.loc[:, "gene_cluster"] = "NA"
    myadata.var.loc[lfc_ad_transpose.obs.index, "gene_cluster"] = lfc_ad_transpose.obs["gene_leiden"].astype(str).values

    # compute cell scores
    cell_score_keys = []
    for g_cluster in myadata.var.loc[:, "gene_cluster"].unique():
        if g_cluster == "NA":
            continue
        sub_ad = lfc_ad[:, myadata.var["gene_cluster"] == g_cluster].X
        signs_ = np.sum(np.sign(sub_ad), 1) / sub_ad.shape[-1]
        score_key = f"score_gene_cluster_{g_cluster}"

        myadata.obs.loc[:, score_key] = signs_.toarray()
        cell_score_keys.append(score_key)
    if plot:
        sc.pp.neighbors(myadata, use_rep=u_repkey)
        sc.tl.umap(myadata)
        sc.pl.umap(myadata, color=cell_score_keys)

    # construct cell clusters from cell scores
    keys_to_plot = []
    for g_cluster in myadata.var.loc[:, "gene_cluster"].unique():
        if g_cluster == "NA":
            continue
        score_key = f"score_gene_cluster_{g_cluster}"
        cell_cluster_key = f"cell_cluster_{g_cluster}"
        cell_extended_cluster_key = f"cell_extended_cluster_{g_cluster}"

        scores = myadata.obs.loc[:, score_key]

        if score_quantile is not None:
            score_threshold = np.quantile(scores, score_quantile)
        is_in_cluster = scores >= score_threshold
        myadata.obs.loc[:, cell_cluster_key] = is_in_cluster
        keys_to_plot.append(cell_cluster_key)

        is_in_extended_cluster = is_in_cluster.copy()
        for _ in range(extend_nhops):
            neighbors = myadata.obsp["connectivities"][is_in_cluster, :].toarray()
            _is_in_extended_cluster = np.any(neighbors, 0)
            is_in_extended_cluster = is_in_extended_cluster | _is_in_extended_cluster
        myadata.obs.loc[:, cell_extended_cluster_key] = is_in_extended_cluster
        keys_to_plot.append(cell_extended_cluster_key)

        myadata.obs.loc[:, cell_cluster_key] = myadata.obs.loc[:, cell_cluster_key].astype(str)
        myadata.obs.loc[:, cell_extended_cluster_key] = myadata.obs.loc[:, cell_extended_cluster_key].astype(str)
    if plot:
        sc.pl.umap(myadata, color=keys_to_plot)


def compute_cell_clusters(
    myadata,
    extend_nhops=2,
    score_threshold=0.5,
    score_quantile=None,
    plot=True,
):
    keys_to_plot = []
    for g_cluster in myadata.var.loc[:, "gene_cluster"].unique():
        if g_cluster == "NA":
            continue
        score_key = f"score_gene_cluster_{g_cluster}"
        cell_cluster_key = f"cell_cluster_{g_cluster}"
        cell_extended_cluster_key = f"cell_extended_cluster_{g_cluster}"

        scores = myadata.obs.loc[:, score_key]

        if score_quantile is not None:
            score_threshold = np.quantile(scores, score_quantile)
        is_in_cluster = scores >= score_threshold
        myadata.obs.loc[:, cell_cluster_key] = is_in_cluster
        keys_to_plot.append(cell_cluster_key)

        is_in_extended_cluster = is_in_cluster.copy()
        for _ in range(extend_nhops):
            neighbors = myadata.obsp["connectivities"][is_in_cluster, :].toarray()
            _is_in_extended_cluster = np.any(neighbors, 0)
            is_in_extended_cluster = is_in_extended_cluster | _is_in_extended_cluster
        myadata.obs.loc[:, cell_extended_cluster_key] = is_in_extended_cluster
        keys_to_plot.append(cell_extended_cluster_key)

        myadata.obs.loc[:, cell_cluster_key] = myadata.obs.loc[:, cell_cluster_key].astype(str)
        myadata.obs.loc[:, cell_extended_cluster_key] = myadata.obs.loc[:, cell_extended_cluster_key].astype(str)
    if plot:
        sc.pl.umap(myadata, color=keys_to_plot)


def fit_deseq2(
    adata,
    design_factors,
):
    counts_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    design_df = adata.obs.loc[:, design_factors].copy()
    dds = DeseqDataSet(
        counts=counts_df,
        clinical=design_df,
        design_factors=design_factors,
        refit_cooks=True,
        n_cpus=24,
    )
    dds.deseq2()
    return dds
