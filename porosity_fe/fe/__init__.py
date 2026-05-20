"""FE subpackage: element, assembler, solver."""

from .assembler import BoundaryHandler, GlobalAssembler
from .element import _NODE_COORDS_REF as _NODE_COORDS_REF  # noqa: F401
from .element import Hex8Element
from .solver import FESolver, FieldResults

__all__ = [
    "BoundaryHandler",
    "FESolver",
    "FieldResults",
    "GlobalAssembler",
    "Hex8Element",
]
