from typing import Any, Tuple, Union

import jax.numpy as jnp
import numpy as np

NdArray = Union[np.ndarray, jnp.ndarray]
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
