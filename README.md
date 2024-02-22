# mrvi

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/workflow/status/justjhong/mrvi/Test/main
[link-tests]: https://github.com/YosefLab/mrvi/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/mrvi

Multi-resolution Variational Inference.

🚧 :warning: This is a soft launch of the new `mrvi` package. We are working on docs and tutorials. :warning: 🚧

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

There are several alternative options to install mrvi:

1. Install the latest pre-release of `mrvi` from `PyPI <https://pypi.org/project/mrvi/>`\_:

```bash
pip install --pre mrvi
```

2. Install the latest development version:

```bash
pip install git+https://github.com/justjhong/mrvi.git@main
```

## User guide

While a more comprehensive user guide is in the works, you can find here a brief overview of the main features of `mrvi`.


**Data preparation**:
MrVI relies on `scvi-tools` routines for model initialization and training.
In particular, `mrvi` assumes data to be stored in an AnnData object.
A first step is to load the data and register it, as follows:

```python
from mrvi import MrVI

MrVI.setup_anndata(adata,  sample_key="my_sample_key", batch_key="my_batch_key")
```
where here `'my_sample_key'` and `'my_batch_key'` are expected to be keys of `adata.obs` that contain the sample and batch assignments, respectively. 


**Model training**:
The next step is to initialize and train the model, which can be done via:

```python
model = MrVI(adata)
model.train()
```

Once the model is trained, we recommend visualizing the validation ELBO to assess convergence, which is stored in `model.history["elbo_validation"]`.
If the ELBO has not converged, you should consider training the model for more epochs.


**Latent space visualization**:
MrVI contains two latent spaces, `u`, that captures global cell-type variations, and `z`, that additionally captures sample-specific variations.
These two latent representations can be accessed via `model.get_latent_representation()`, (with `give_z=True` to access `z`).
In particular, these latent variables can be seemlessly used for data visualization or clustering using scanpy.
For instance, visualizing the `u` latent space can be done via:

```python
import scanpy as sc
from scvi.model.utils import mde

u = model.get_latent_representation()
u_mde = mde(u)
adata.obsm["u_mde"] = u_mde
sc.pl.embedding(adata, basis="u_mde")
```


**Computing sample-sample dissimilarities**:
MrVI can be used to predict sample-sample dissimilarities, using the following snippet:

```python
# Predict sample-sample dissimilarities per cell type
dists = model.get_local_sample_distances(
    adata, keep_cell=False, groupby="initial_clustering", batch_size=32
)

# OR predict sample-sample dissimilarities for EACH cell
# WARNING: this can be slow and memory-intensive for large datasets
dists = model.get_local_sample_distances(
    adata, keep_cell=True, batch_size=32
)
```
These dissimilarities can then be visualized via `seaborn.clustermap` or similar tools.


**DE analysis**: MrVI can be used to identify differentially expressed genes (DEGs) between two groups of samples at the single-cell level.
Here, "samples" refere to the `sample_key` provided in `MrVI.setup_anndata`.
Identifying such genes can be done as follows,

```python
donor_keys_ = ["Status"]  # Here, Status is the donor covarate of interest
multivariate_analysis_kwargs = {
    "batch_size": 128,
    "normalize_design_matrix": True,
    "offset_design_matrix": False,
    "store_lfc": True,
    "eps_lfc": 1e-4,
}
res = model.perform_multivariate_analysis(
    donor_keys=donor_keys_,
    donor_subset=donor_subset,
    **multivariate_analysis_kwargs,
)
```

**DA analysis**:
MrVI can also be used to assess differences in cell-type compositions across sample groups, using the following snippet:

```python
da_res = model.get_outlier_cell_sample_pairs()
gp1 = model.donor_info.query('Status == "A"').patient_id.values
gp2 = model.donor_info.query('Status == "B"').patient_id.values
log_p1 = da_res.log_probs.loc[{"sample": gp1}]
log_p1 = logsumexp(log_p1, axis=1) - np.log(log_p1.shape[1])
log_p2 = da_res.log_probs.loc[{"sample": gp2}]
log_p2 = logsumexp(log_p2, axis=1) - np.log(log_p2.shape[1])
log_prob_ratio = log_p1 - log_p2
```



## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/justjhong/mrvi/issues
[changelog]: https://mrvi.readthedocs.io/latest/changelog.html
[link-docs]: https://mrvi.readthedocs.io
[link-api]: https://mrvi.readthedocs.io/latest/api.html
