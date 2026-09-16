"""Extraction methods behind one interface (``contracts/extraction-method.md``).

Every method — prototype adapter, trivial baselines, the POC — is scored by the identical harness,
so a method comparison compares methods, never harness variants. Registry in ``base``.
"""

from hsreloc.extraction.methods.base import get_method, registered_methods  # noqa: F401

# Importing the implementations registers them (base.register decorator).
from hsreloc.extraction.methods import baseline_gradient  # noqa: E402,F401
from hsreloc.extraction.methods import baseline_threshold  # noqa: E402,F401
from hsreloc.extraction.methods import poc_robust_dp  # noqa: E402,F401
from hsreloc.extraction.methods import prototype_dp  # noqa: E402,F401
