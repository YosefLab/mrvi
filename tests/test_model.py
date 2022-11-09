import numpy as np
from scvi.data import synthetic_iid

from scvi_v2 import MrVI


def test_mrvi():
    adata = synthetic_iid()
    adata.obs["sample"] = np.random.choice(15, size=adata.shape[0])
    MrVI.setup_anndata(adata, sample_key="sample", categorical_nuisance_keys=["batch"])
    model = MrVI(
        adata,
        n_latent_sample=5,
    )
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.is_trained_ = True
    model.history
    assert model.get_latent_representation().shape == (adata.shape[0], 10)
    assert model.get_latent_representation(use_mean=False, mc_samples=2).shape == (adata.shape[0], 10)
    assert model.get_latent_representation(use_mean=False, mc_samples=1).shape == (adata.shape[0], 10)
    assert model.get_local_sample_representation().shape == (adata.shape[0], 15, 10)
    assert model.get_local_sample_representation(return_distances=True).shape == (
        adata.shape[0],
        15,
        15,
    )
    # tests __repr__
    print(model)
