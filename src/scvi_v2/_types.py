from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional, Tuple, TypeVar, Union

import jax.numpy as jnp
import numpy as np
import xarray as xr

NdArray = Union[np.ndarray, jnp.ndarray]
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
MrVI = TypeVar("MrVI")


@dataclass(frozen=True)
class MrVIReduction:
    """
    A dataclass object that represents a single reduction for ``MrVI.compute_local_statistics``.

    Parameters
    ----------
    name
        Name of the reduction. Used as the key for the corresponding DataArray.
    input
        Type of input data.
    fn
        Function that computes the reduction.
    map_by
        Covariate name that parameterizes the ``fn``. If provided, ``fn`` must
        accept a third argument, ``map_by``.
    group_by
        Covariate name by which to average the computed statistics by. If ``None``,
        the outputs are left at the per-cell granularity.
    """

    name: str
    input: Union[
        Literal["mean_representations"],
        Literal["mean_distances"],
        Literal["sampled_representations"],
        Literal["sampled_distances"],
        Literal["normalized_distances"],
    ]
    fn: Union[
        Callable[[MrVI, xr.DataArray], xr.DataArray], Callable[[MrVI, xr.DataArray, str], xr.DataArray]
    ] = lambda _, x: xr.DataArray(x)
    map_by: Optional[str] = None
    group_by: Optional[str] = None


@dataclass(frozen=True)
class ComputeLocalStatisticsConfig:
    """
    A configuration object for ``MrVI.compute_local_statistics``.

    Parameters
    ----------
        reductions
            List of `MrVIReduction` objects defining values to compute over each cell.
    """

    reductions: Iterable[MrVIReduction] = (MrVIReduction(name="mean_representations", input="mean_representations"),)


@dataclass(frozen=True)
class _ComputeLocalStatisticsRequirements:
    """Utility class for the summarized requirements for ``MrVI.compute_local_statistics``."""

    needs_mean_representations: bool
    needs_mean_distances: bool
    needs_sampled_representations: bool
    needs_sampled_distances: bool
    needs_normalized_distances: bool
    ungrouped_reductions: Iterable[MrVIReduction]
    grouped_reductions: Iterable[MrVIReduction]
