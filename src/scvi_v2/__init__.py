from importlib.metadata import version

from ._model import MrVI
from ._module import MrVAE
from ._types import MrVIReduction
from ._utils import permutation_test

__all__ = ["MrVI", "MrVAE", "MrVIReduction", "DecoderZX", "DecoderUZ", "EncoderXU", "permutation_test"]

__version__ = version("scvi-v2")
