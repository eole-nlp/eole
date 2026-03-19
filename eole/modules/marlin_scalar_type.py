"""
eole/modules/marlin_scalar_type.py

Scalar type definitions for Marlin quantization.
The ``id`` values must match
``eole/csrc/quantization/marlin/eole_scalar_type.hpp::ScalarType::from_id()``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarType:
    """A quantization scalar type used by the Marlin CUDA kernels.

    The ``id`` field is the integer used as ``b_q_type_id`` in eole._ops
    kernel calls.  It must match the switch-case in eole_scalar_type.hpp.
    """

    _id: int  # simple eole kernel ID (must match eole_scalar_type.hpp)
    size_bits: int  # total bits per element

    @property
    def id(self) -> int:
        """Simple integer type ID expected by eole._ops kernels."""
        return self._id

    def __str__(self) -> str:
        return f"ScalarType(id={self._id}, bits={self.size_bits})"


class _ScalarTypes:
    """Named scalar types used by Marlin kernels.

    IDs must match ``eole/csrc/quantization/marlin/eole_scalar_type.hpp``::

        kU4B8   = {  4, 4, false, "uint4b8"  }   ← symmetric int4
        kU8B128 = {  5, 8, false, "uint8b128" }   ← symmetric int8
    """

    uint4b8 = ScalarType(_id=4, size_bits=4)  # symmetric 4-bit
    uint8b128 = ScalarType(_id=5, size_bits=8)  # symmetric 8-bit


scalar_types = _ScalarTypes()

__all__ = ["ScalarType", "scalar_types"]
