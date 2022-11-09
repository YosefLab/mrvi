# %%
import scanpy as sc
from scvi_v2 import MrVI
# %%
adata = sc.read_h5ad("/home/pierre/largen_experiments/workflow/data/haniffa/adata.processed.h5ad")
# %%
MrVI.setup_anndata(adata, sample_key="sample_id", categorical_nuisance_keys=["Site"])
# %%


model = MrVI(adata)
model.train(
    # max_epochs=200,
    # check_val_every_n_epoch=1,
    # train_size=0.5,
    # early_stopping=True,
    # early_stopping_patience=15,
)
# %%
