import numpy as np
from mrvi import MrVI
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
    model = MrVI(adata)
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
    model = MrVI(adata)
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
    model = MrVI(adata)
    model.train(2, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()
    model.get_outlier_cell_sample_pairs(subsample_size=50)
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
    model = MrVI(adata)
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
        scale_observations=True,
        qz_kwargs={
            "use_map": False,
        },
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()
    model.get_local_sample_distances(normalize_distances=True)

    model = MrVI(
        adata,
        n_latent=n_latent,
        scale_observations=True,
        qz_kwargs={
            "use_map": False,
        },
        px_kwargs={"low_dim_batch": False},
        u_prior_mixture=True,
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()
    model.get_local_sample_distances(normalize_distances=True)

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
        z_u_prior=False,
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()
    model.get_local_sample_distances(normalize_distances=True)

    model = MrVI(
        adata,
        n_latent=n_latent,
        scale_observations=True,
        qz_kwargs={
            "use_map": False,
        },
        px_kwargs={"low_dim_batch": True},
        learn_z_u_prior_scale=True,
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances()


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
        scale_observations=True,
        qz_kwargs={
            "use_map": False,
        },
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

    model = MrVI(
        adata,
        n_latent=n_latent,
        qz_kwargs={"use_map": True},
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_local_sample_distances(use_mean=True)
    model.get_local_sample_distances()

    model = MrVI(
        adata,
        n_latent=n_latent,
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
