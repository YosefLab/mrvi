import warnings
from importlib.metadata import version

from ._model import MrVI
from ._module import MrVAE
from ._types import MrVIReduction

warnings.warn(
    "This package is deprecated. For the latest version of MrVI, please install `scvi-tools` and import the model class via `scvi.external.MRVI`.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "MrVI",
    "MrVAE",
    "MrVIReduction",
    "DecoderZX",
    "DecoderUZ",
    "EncoderXU",
]

__version__ = version("mrvi")
