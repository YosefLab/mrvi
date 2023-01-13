import numpy as np
from scvi.data import synthetic_iid

from scvi_v2 import MrVI


def test_mrvi():
    adata = synthetic_iid()
    adata.obs["sample"] = np.random.choice(15, size=adata.shape[0])
    meta1 = np.random.randint(0, 2, size=15)
    adata.obs["meta1"] = meta1[adata.obs["sample"].values]

    meta2 = np.random.randn(15)
    adata.obs["meta2"] = meta2[adata.obs["sample"].values]
    MrVI.setup_anndata(adata, sample_key="sample", batch_key="batch")
    adata.obs["cont_cov"] = np.random.normal(0, 1, size=adata.shape[0])
    MrVI.setup_anndata(adata, sample_key="sample", batch_key="batch", continuous_covariate_keys=["cont_cov"])
    model = MrVI(
        adata,
        n_latent_sample=5,
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.is_trained_ = True
    model.history
    assert model.get_latent_representation().shape == (adata.shape[0], 10)
    local_vmap = model.get_local_sample_representation()
    assert local_vmap.shape == (adata.shape[0], 15, 10)
    local_dist_vmap = model.get_local_sample_representation(return_distances=True)
    assert local_dist_vmap.shape == (
        adata.shape[0],
        15,
        15,
    )
    local_map = model.get_local_sample_representation(use_vmap=False)
    local_dist_map = model.get_local_sample_representation(return_distances=True, use_vmap=False)
    assert local_map.shape == (adata.shape[0], 15, 10)
    assert local_dist_map.shape == (
        adata.shape[0],
        15,
        15,
    )
    assert np.allclose(local_map, local_vmap, atol=1e-6)
    assert np.allclose(local_dist_map, local_dist_vmap, atol=1e-6)

    donor_keys = [
        ("meta1", "nn"),
        ("meta2", "geary"),
    ]
    pvals = model.compute_cell_scores(donor_keys=donor_keys)
    assert pvals.shape == (adata.shape[0], 2)
    es = model.compute_cell_scores(donor_keys=donor_keys, compute_pval=False)
    assert es.shape == (adata.shape[0], 2)

    # tests __repr__
    print(model)
