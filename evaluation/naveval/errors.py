"""Typed errors for the evaluator's schema and contract validation.

Malformed input is an error, not a warning (contracts/run-record.md, contracts/dataset.md):
silently analysing a truncated run or a convention violation would produce a plausible
number for data that does not mean what it appears to mean.
"""

from __future__ import annotations


class NavevalError(Exception):
    """Base class for all naveval errors."""


class SchemaVersionError(NavevalError):
    """A file declares a schema major version this reader does not support."""


class ContractViolationError(NavevalError):
    """A file violates one of its format's stated invariants."""


class ConventionError(NavevalError):
    """A dataset declares a coordinate or heading convention other than canonical."""


class DatasetMismatchError(NavevalError):
    """A run record's dataset_id/revision does not match the dataset it is evaluated against."""
