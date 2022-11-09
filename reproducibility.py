# %%
import itertools
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from mrvi import MrVI as MrVITorch
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from scvi_v2 import MrVI as MrVIJAX
import plotnine as p9
import seaborn as sns
from matplotlib.cm import tab10

# from scipy.hierarchy import linkage
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

METRIC = "euclidean"


def compute_aggregate_dmat(reps):
    # return pairwise_distances(reps.mean(0))
    n_cells, n_donors, _ = reps.shape
    pairwise_ds = np.zeros((n_donors, n_donors))
    for x in tqdm(reps):
        d_ = pairwise_distances(x, metric=METRIC)
        pairwise_ds += d_ / n_cells
    return pairwise_ds


def compute_agg_dmats(model):
    rep = model.get_local_sample_representation(adata=None, return_distances=False)
    res = []
    for cluster in adata.obs["celltype"].unique():
        good_cells = adata.obs["celltype"] == cluster
        rep_ct = rep[good_cells]
        ss_matrix = compute_aggregate_dmat(rep_ct)
        res.append(
            pd.DataFrame(dict(dist=ss_matrix.reshape(-1), pair_idx=np.arange(ss_matrix.size),)).assign(
                cluster=cluster,
            )
        )
    return pd.concat(res, 0)


# %%
def make_categorical(adata, obs_key):
    adata.obs[obs_key] = adata.obs[obs_key].astype("category")


def assign_symsim_donors(adata):
    np.random.seed(1)
    donor_key = "donor"
    batch_key = "batch"

    n_donors = 32
    n_meta = len([k for k in adata.obs.keys() if "meta_" in k])
    meta_keys = [f"meta_{i + 1}" for i in range(n_meta)]
    make_categorical(adata, batch_key)
    batches = adata.obs[batch_key].cat.categories.tolist()
    n_batch = len(batches)

    meta_combos = list(itertools.product([0, 1], repeat=n_meta))
    donors_per_meta_batch_combo = n_donors // len(meta_combos) // n_batch

    # Assign donors uniformly at random for cells with matching metadata.
    donor_assignment = np.empty(adata.n_obs, dtype=object)
    for batch in batches:
        batch_donors = []
        for meta_combo in meta_combos:
            match_cats = [f"CT{meta_combo[i]+1}:1" for i in range(n_meta)]
            eligible_cell_idxs = (
                (
                    np.all(
                        adata.obs[meta_keys].values == match_cats,
                        axis=1,
                    )
                    & (adata.obs[batch_key] == batch)
                )
                .to_numpy()
                .nonzero()[0]
            )
            meta_donors = [
                f"donor_meta{meta_combo}_batch{batch}_{ch}"
                for ch in string.ascii_lowercase[:donors_per_meta_batch_combo]
            ]
            donor_assignment[eligible_cell_idxs] = np.random.choice(
                meta_donors, replace=True, size=len(eligible_cell_idxs)
            )
            batch_donors += meta_donors

    adata.obs[donor_key] = donor_assignment

    donor_meta = adata.obs[donor_key].str.extractall(r"donor_meta\(([0-1]), ([0-1]), ([0-1])\)_batch[0-9]_[a-z]")
    for match_idx, meta_key in enumerate(meta_keys):
        adata.obs[f"donor_{meta_key}"] = donor_meta[match_idx].astype(int).tolist()


adata = sc.read_h5ad("/home/pierre/data/symsim_new.h5ad")
assign_symsim_donors(adata)
# %%
model_kwargs = dict(
    n_latent=10,
    n_latent_sample=2,
    # uz_scaler=False,
    # encoder_n_hidden=128,
    px_kwargs=None,
    pz_kwargs=None,
)

train_kwargs = dict(
    max_epochs=200,
    early_stopping=True,
    early_stopping_patience=15,
    check_val_every_n_epoch=1,
    batch_size=256,
    train_size=0.9,
    plan_kwargs={"lr": 1e-2, "n_epochs_kl_warmup": 20},
)

# %%
MrVIJAX.setup_anndata(adata, sample_key="donor", categorical_nuisance_keys=["batch"])
adata2 = adata.copy()
MrVITorch.setup_anndata(adata2, sample_key="donor", categorical_nuisance_keys=["batch"])
all_res = []

for seed in [0, 1]:
    model1 = MrVIJAX(adata, **model_kwargs)
    model1.train(**train_kwargs)
    rep1 = model1.get_local_sample_representation(adata=None, return_distances=False)
    res_ = compute_agg_dmats(model1)
    res_ = res_.assign(seed=seed, model="jax")
    all_res.append(res_)

    model2 = MrVITorch(adata2, **model_kwargs)
    model2.train(**train_kwargs)
    rep2 = model2.get_local_sample_representation(adata=None, return_distances=False)
    res_ = compute_agg_dmats(model2)
    res_ = res_.assign(seed=seed, model="torch")
    all_res.append(res_)

all_res = pd.concat(all_res, 0)

# %%
ax = model1.history["train_loss_step"].plot()
model2.history["train_loss_step"].plot(ax=ax)
# %%
# all_res
# %%
v2 = (
    (
        all_res.loc[lambda x: x.cluster == "CT1:1"].groupby(["cluster", "seed"]).apply(
            lambda x: pearsonr(x.loc[lambda x: x.model == "jax", "dist"], x.loc[lambda x: x.model == "torch", "dist"])[
                0
            ]
        )
    )
    .to_frame("correlation")
    .assign(experiment="variability across implementations")
)
v2
# %%
v1 = (
    (
        all_res.loc[lambda x: x.cluster == "CT1:1"].loc[lambda x: x.model == "torch"]
        .groupby("cluster")
        .apply(lambda x: pearsonr(x.loc[lambda x: x.seed == 0, "dist"], x.loc[lambda x: x.seed == 1, "dist"])[0])
    )
    .to_frame("correlation")
    .assign(experiment="variability across seeds")
)
# %%


(p9.ggplot(pd.concat([v1, v2], 0), p9.aes(x="experiment", y="correlation")) + p9.geom_point())
# %%


# %%
meta = adata.obs.loc[lambda x: ~x._scvi_sample.duplicated(keep="first")].set_index("_scvi_sample").sort_index()
colors = meta[["meta_1", "meta_2", "meta_3"]].apply(lambda x: x.cat.codes, 0).applymap(lambda x: tab10(x))
# %%


# %%
for model in [model1, model2]:
# for model in [model1]:
    rep = model.get_local_sample_representation(adata=None, return_distances=False)
    for cluster in adata.obs["celltype"].unique():
        good_cells = adata.obs["celltype"] == cluster
        rep_ct = rep[good_cells]
        ss_matrix = compute_aggregate_dmat(rep_ct)
        p = squareform(ss_matrix)
        Z = hierarchy.linkage(p, method="complete")
        Z = hierarchy.optimal_leaf_ordering(Z, p)
        ss_matrix = pd.DataFrame(ss_matrix)

        sns.clustermap(
            ss_matrix, row_linkage=Z, col_linkage=Z, cmap="viridis", figsize=(10, 10),
            row_colors=colors)
        plt.show()
# %%
# %%
