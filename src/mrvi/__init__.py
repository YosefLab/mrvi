from importlib.metadata import version

from ._model import MrVI
from ._module import MrVAE
from ._types import MrVIReduction

__all__ = [
    "MrVI",
    "MrVAE",
    "MrVIReduction",
    "DecoderZX",
    "DecoderUZ",
    "EncoderXU",
]

__version__ = version("mrvi")
