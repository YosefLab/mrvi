from copy import deepcopy
from tempfile import TemporaryDirectory

import flax.linen as nn
import numpy as np
from mrvi import MrVI
import jax
from scvi.data import synthetic_iid


def test_mrvi():
    adata = synthetic_iid()
    adata.obs["sample"] = np.random.choice(15, size=adata.shape[0])
    adata.obs["sample_str"] = "sample_" + adata.obs["sample"].astype(str)
    meta1 = np.random.randint(0, 2, size=15)
    adata.obs["meta1"] = meta1[adata.obs["sample"].values]

    meta2 = np.random.randn(15)
    adata.obs["meta2"] = meta2[adata.obs["sample"].values]
    MrVI.setup_anndata(adata, sample_key="sample", batch_key="batch")
    adata.obs["cont_cov"] = np.random.normal(0, 1, size=adata.shape[0])
    n_latent = 10

    adata.obs["meta1_cat"] = "CAT_" + adata.obs["meta1"].astype(str)
    adata.obs["meta1_cat"] = adata.obs["meta1_cat"].astype("category")

    adata.obs.loc[:, "disjoint_batch"] = (adata.obs.loc[:, "sample"] <= 6).replace(
        {True: "batch_0", False: "batch_1"}
    )
    MrVI.setup_anndata(adata, sample_key="sample", batch_key="disjoint_batch")
    model = MrVI(
        adata,
        px_nn_flavor="attention",
        qz_nn_flavor="attention",
    )
    model.train(10, check_val_every_n_epoch=1, train_size=0.5)
    donor_keys = ["meta1_cat", "meta2", "cont_cov"]
    model.perform_multivariate_analysis(
        donor_keys=donor_keys, store_lfc=True, add_batch_specific_offsets=True
    )
    model.perform_multivariate_analysis(
        donor_keys=donor_keys,
        store_lfc=True,
        lambd=1e-1,
        add_batch_specific_offsets=True,
    )
    model.perform_multivariate_analysis(
        donor_keys=donor_keys,
        store_lfc=True,
        filter_donors=True,
        add_batch_specific_offsets=True,
    )
    model.get_local_sample_distances()

    MrVI.setup_anndata(adata, sample_key="sample", batch_key="batch")
    model = MrVI(
        adata,
        px_nn_flavor="attention",
        qz_nn_flavor="attention",
    )
    model.train(2, check_val_every_n_epoch=1, train_size=0.5)
    donor_keys = ["meta1_cat", "meta2", "cont_cov"]
    model.perform_multivariate_analysis(
        donor_keys=donor_keys, store_lfc=True, add_batch_specific_offsets=False
    )
    model.get_local_sample_distances()

    MrVI.setup_anndata(
        adata,
        sample_key="sample",
        batch_key="batch",
        continuous_covariate_keys=["cont_cov"],
    )
    model = MrVI(
        adata,
        px_nn_flavor="attention",
        qz_nn_flavor="attention",
    )
    model.train(2, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()
    model.get_outlier_cell_sample_pairs(flavor="ball", subsample_size=50)
    model.get_outlier_cell_sample_pairs(flavor="MoG", subsample_size=50)
    model.get_outlier_cell_sample_pairs(flavor="ap", subsample_size=50)
    model.perform_multivariate_analysis(
        donor_keys=donor_keys, store_lfc=True, add_batch_specific_offsets=False
    )

    adata.obs.loc[:, "batch_placeholder"] = "1"
    MrVI.setup_anndata(adata, sample_key="sample", batch_key="batch_placeholder")
    model = MrVI(adata)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.perform_multivariate_analysis(donor_keys=donor_keys, store_lfc=True)
    model.perform_multivariate_analysis(
        donor_keys=donor_keys, store_lfc=True, lambd=1e-1
    )
    model.perform_multivariate_analysis(
        donor_keys=donor_keys, store_lfc=True, filter_donors=True
    )

    MrVI.setup_anndata(
        adata,
        sample_key="sample_str",
        batch_key="batch",
        continuous_covariate_keys=["cont_cov"],
    )
    model = MrVI(
        adata,
        px_nn_flavor="attention",
        qz_nn_flavor="attention",
    )
    model.train(2, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()
    donor_keys = ["meta1_cat", "meta2", "cont_cov"]
    donor_subset = [f"sample_{i}" for i in range(8)]
    model.perform_multivariate_analysis(donor_keys=donor_keys, donor_subset=donor_subset)

    MrVI.setup_anndata(
        adata,
        sample_key="sample",
        batch_key="batch",
        continuous_covariate_keys=["cont_cov"],
    )
    model = MrVI(
        adata,
        n_latent=n_latent,
        laplace_scale=1.0,
        qz_nn_flavor="linear",
        qz_kwargs={"n_factorized_embed_dims": 3},
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)

    model = MrVI(
        adata,
        n_latent=n_latent,
        scale_observations=True,
        qz_nn_flavor="linear",
        qz_kwargs={"n_factorized_embed_dims": 3},
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)

    model = MrVI(
        adata,
        n_latent=n_latent,
        scale_observations=True,
        qz_kwargs={"use_map": False, "stop_gradients": True},
        qz_nn_flavor="mlp",
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        scale_observations=True,
        qz_kwargs={
            "use_map": False,
        },
        qz_nn_flavor="attention",
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        scale_observations=True,
        qz_kwargs={
            "use_map": False,
        },
        px_kwargs={"low_dim_batch": False},
        u_prior_mixture=True,
        px_nn_flavor="attention",
        qz_nn_flavor="attention",
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        scale_observations=True,
        qz_kwargs={
            "use_map": False,
            "stop_gradients": False,
            "stop_gradients_mlp": True,
        },
        px_kwargs={
            "low_dim_batch": False,
            "stop_gradients": False,
            "stop_gradients_mlp": True,
        },
        px_nn_flavor="attention",
        qz_nn_flavor="attention",
        z_u_prior=False,
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        scale_observations=True,
        qz_kwargs={
            "use_map": False,
        },
        px_kwargs={"low_dim_batch": True},
        px_nn_flavor="attention",
        qz_nn_flavor="attention",
        learn_z_u_prior_scale=True,
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    # model = MrVI(
    #     adata,
    #     n_latent=n_latent,
    #     qz_nn_flavor="linear",
    #     qz_kwargs={"use_nonlinear": True},
    # )
    # model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    # model.is_trained_ = True
    # _ = model.history

    # assert model.get_latent_representation().shape == (adata.shape[0], n_latent)
    # local_vmap = model.get_local_sample_representation()

    # assert local_vmap.shape == (adata.shape[0], 15, n_latent)
    # local_dist_vmap = model.get_local_sample_distances()["cell"]
    # assert local_dist_vmap.shape == (
    #     adata.shape[0],
    #     15,
    #     15,
    # )
    # local_map = model.get_local_sample_representation(use_vmap=False)
    # model.get_local_sample_distances(use_vmap=False)["cell"]
    # model.get_local_sample_distances(use_vmap=False, norm="l1")["cell"]
    # model.get_local_sample_distances(use_vmap=False, norm="linf")["cell"]
    # local_dist_map = model.get_local_sample_distances(use_vmap=False, norm="l2")["cell"]
    # assert local_map.shape == (adata.shape[0], 15, n_latent)
    # assert local_dist_map.shape == (
    #     adata.shape[0],
    #     15,
    #     15,
    # )
    # assert np.allclose(local_map, local_vmap, atol=1e-3)
    # assert np.allclose(local_dist_map, local_dist_vmap, atol=1e-3)

    # local_normalized_dists = model.get_local_sample_distances()[
    #     "cell"
    # ]
    # assert local_normalized_dists.shape == (
    #     adata.shape[0],
    #     15,
    #     15,
    # )
    # assert np.allclose(
    #     local_normalized_dists[0].values, local_normalized_dists[0].values.T, atol=1e-6
    # )

    # # Test memory efficient groupby.
    # model.get_local_sample_distances(keep_cell=False, groupby=["meta1", "meta2"])
    # grouped_dists_no_cell = model.get_local_sample_distances(
    #     keep_cell=False, groupby=["meta1", "meta2"]
    # )
    # grouped_dists_w_cell = model.get_local_sample_distances(groupby=["meta1", "meta2"])
    # assert np.allclose(grouped_dists_no_cell.meta1, grouped_dists_w_cell.meta1)
    # assert np.allclose(grouped_dists_no_cell.meta2, grouped_dists_w_cell.meta2)

    # grouped_normalized_dists = model.get_local_sample_distances(
    #     , keep_cell=False, groupby=["meta1", "meta2"]
    # )
    # assert grouped_normalized_dists.meta1.shape == (
    #     2,
    #     15,
    #     15,
    # )

    # # tests __repr__
    # print(model)


def test_mrvi_shrink_u():
    adata = synthetic_iid()
    adata.obs["sample"] = np.random.choice(15, size=adata.shape[0])
    meta1 = np.random.randint(0, 2, size=15)
    adata.obs["meta1"] = meta1[adata.obs["sample"].values]

    meta2 = np.random.randn(15)
    adata.obs["meta2"] = meta2[adata.obs["sample"].values]
    MrVI.setup_anndata(adata, sample_key="sample", batch_key="batch")
    adata.obs["cont_cov"] = np.random.normal(0, 1, size=adata.shape[0])
    MrVI.setup_anndata(
        adata,
        sample_key="sample",
        batch_key="batch",
        continuous_covariate_keys=["cont_cov"],
    )
    n_latent_u = 5
    n_latent = 10

    model = MrVI(
        adata,
        n_latent=n_latent,
        n_latent_u=n_latent_u,
        laplace_scale=1.0,
        qz_nn_flavor="linear",
        qz_kwargs={"n_factorized_embed_dims": 3},
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        n_latent_u=n_latent_u,
        laplace_scale=1.0,
        qz_nn_flavor="linear",
        qz_kwargs={"n_factorized_embed_dims": 3},
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        n_latent_u=n_latent_u,
        scale_observations=True,
        qz_nn_flavor="linear",
        qz_kwargs={"n_factorized_embed_dims": 3},
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        n_latent_u=n_latent_u,
        scale_observations=True,
        qz_kwargs={"use_map": False, "stop_gradients": True},
        qz_nn_flavor="mlp",
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        n_latent_u=n_latent_u,
        scale_observations=True,
        qz_kwargs={
            "use_map": False,
        },
        qz_nn_flavor="attention",
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        n_latent_u=n_latent_u,
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.is_trained_ = True
    _ = model.history

    assert model.get_latent_representation().shape == (adata.shape[0], n_latent_u)


def test_mrvi_stratifications():
    adata = synthetic_iid()
    adata.obs["sample"] = np.random.choice(15, size=adata.shape[0])
    meta1 = np.random.randint(0, 2, size=15)
    adata.obs["meta1"] = meta1[adata.obs["sample"].values]

    meta2 = np.random.randn(15)
    adata.obs["meta2"] = meta2[adata.obs["sample"].values]
    MrVI.setup_anndata(adata, sample_key="sample", batch_key="batch")
    adata.obs["cont_cov"] = np.random.normal(0, 1, size=adata.shape[0])
    MrVI.setup_anndata(
        adata,
        sample_key="sample",
        batch_key="batch",
        continuous_covariate_keys=["cont_cov"],
    )
    n_latent = 10
    model = MrVI(
        adata,
        n_latent=n_latent,
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.is_trained_ = True
    _ = model.history

    adata.obs.loc[:, "label_2"] = np.random.choice(2, size=adata.shape[0])
    dists = model.get_local_sample_distances(groupby=["labels", "label_2"])
    cell_dists = dists["cell"]
    assert cell_dists.shape == (
        adata.shape[0],
        15,
        15,
    )
    ct_dists = dists["labels"]
    assert ct_dists.shape == (
        3,
        15,
        15,
    )
    assert np.allclose(ct_dists[0].values, ct_dists[0].values.T, atol=1e-6)
    ct_dists = dists["label_2"]
    assert ct_dists.shape == (
        2,
        15,
        15,
    )
    assert np.allclose(ct_dists[0].values, ct_dists[0].values.T, atol=1e-6)

    with TemporaryDirectory() as d:
        model.explore_stratifications(
            dists["labels"], sample_metadata="meta1", figure_dir=d
        )
    model.explore_stratifications(
        dists["labels"], sample_metadata="meta1", show_figures=True
    )
    model.explore_stratifications(dists["labels"], cell_type_keys="label_0")
    model.explore_stratifications(dists["labels"], cell_type_keys=["label_0", "label_1"])

    donor_keys = [
        ("meta1", "nn"),
        ("meta2", "geary"),
    ]
    pvals = model.compute_cell_scores(donor_keys=donor_keys)
    assert len(pvals.data_vars) == 2
    assert pvals.data_vars["meta1_nn_pval"].shape == (adata.shape[0],)
    assert pvals.data_vars["meta2_geary_pval"].shape == (adata.shape[0],)
    es = model.compute_cell_scores(donor_keys=donor_keys, compute_pval=False)
    assert len(es.data_vars) == 2
    assert es.data_vars["meta1_nn_effect_size"].shape == (adata.shape[0],)
    assert es.data_vars["meta2_geary_effect_size"].shape == (adata.shape[0],)


def test_mrvi_nonlinear():
    adata = synthetic_iid()
    adata.obs["sample"] = np.random.choice(15, size=adata.shape[0])
    meta1 = np.random.randint(0, 2, size=15)
    adata.obs["meta1"] = meta1[adata.obs["sample"].values]

    meta2 = np.random.randn(15)
    adata.obs["meta2"] = meta2[adata.obs["sample"].values]
    MrVI.setup_anndata(adata, sample_key="sample", batch_key="batch")
    adata.obs["cont_cov"] = np.random.normal(0, 1, size=adata.shape[0])
    MrVI.setup_anndata(
        adata,
        sample_key="sample",
        batch_key="batch",
        continuous_covariate_keys=["cont_cov"],
    )

    n_latent = 10
    # model = MrVI(
    #     adata,
    #     n_latent=n_latent,
    #     qz_nn_flavor="linear",
    #     qz_kwargs={"use_nonlinear": True},
    # )
    # model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    # model.is_trained_ = True
    # _ = model.history
    # assert model.get_latent_representation().shape == (adata.shape[0], n_latent)
    # local_vmap = model.get_local_sample_representation()

    # assert local_vmap.shape == (adata.shape[0], 15, n_latent)
    # local_dist_vmap = model.get_local_sample_distances()["cell"]
    # assert local_dist_vmap.shape == (
    #     adata.shape[0],
    #     15,
    #     15,
    # )

    # local_normalized_dists = model.get_local_sample_distances(normalize_distances=True)[
    #     "cell"
    # ]
    # assert local_normalized_dists.shape == (
    #     adata.shape[0],
    #     15,
    #     15,
    # )
    # assert np.allclose(
    #     local_normalized_dists[0].values, local_normalized_dists[0].values.T, atol=1e-6
    # )

    model = MrVI(
        adata,
        n_latent=n_latent,
        qz_nn_flavor="mlp",
        qz_kwargs={"use_map": True},
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        qz_nn_flavor="mlp",
        qz_kwargs={"use_map": False},
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        qz_nn_flavor="attention",
        qz_kwargs={"use_map": True},
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances(use_mean=True)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
        qz_nn_flavor="attention",
        qz_kwargs={"use_map": False},
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_latent_representation()
    model.get_local_sample_distances(use_mean=True)
    model.get_local_sample_distances(use_mean=False)
    model.get_local_sample_distances()


def test_compute_local_statistics():
    adata = synthetic_iid()
    n_sample = 15
    adata.obs["sample"] = np.random.choice(n_sample, size=adata.shape[0])
    meta1 = np.random.randint(0, 2, size=n_sample)
    adata.obs["meta1"] = meta1[adata.obs["sample"].values]
    MrVI.setup_anndata(adata, sample_key="sample", batch_key="batch")
    # n_latent = 10
    # model = MrVI(
    #     adata,
    #     n_latent=n_latent,
    #     qz_nn_flavor="linear",
    #     qz_kwargs={"use_nonlinear": True},
    # )
    # model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    # model.is_trained_ = True
    # _ = model.history

    # reductions = [
    #     MrVIReduction(
    #         name="test1",
    #         input="mean_representations",
    #         fn=lambda x: x,
    #         group_by=None,
    #     ),
    #     MrVIReduction(
    #         name="test2",
    #         input="sampled_representations",
    #         fn=lambda x: x + 2,
    #         group_by="meta1",
    #     ),
    #     MrVIReduction(
    #         name="test3",
    #         input="normalized_distances",
    #         fn=lambda x: x + 3,
    #         group_by="meta1",
    #     ),
    # ]
    # outs = model.compute_local_statistics(reductions, mc_samples=10)
    # assert len(outs.data_vars) == 3
    # assert outs["test1"].shape == (adata.shape[0], n_sample, n_latent)
    # assert outs["test2"].shape == (2, 10, n_sample, n_latent)
    # assert outs["test3"].shape == (2, n_sample, n_sample)

    # adata2 = synthetic_iid()
    # adata2.obs["sample"] = np.random.choice(15, size=adata.shape[0])
    # meta1_2 = np.random.randint(0, 2, size=15)
    # adata2.obs["meta1"] = meta1_2[adata2.obs["sample"].values]
    # outs2 = model.compute_local_statistics(reductions, adata=adata2, mc_samples=10)
    # assert len(outs2.data_vars) == 3
    # assert outs2["test1"].shape == (adata2.shape[0], n_sample, n_latent)
    # assert outs2["test2"].shape == (2, 10, n_sample, n_latent)
    # assert outs2["test3"].shape == (2, n_sample, n_sample)
