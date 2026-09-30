# Deprecated module name: FullModel now lives in quantum_electron.full_model.
# This module is kept so that existing imports (and pickled FullModel objects) keep working.
import warnings
from .full_model import FullModel  # noqa: F401

warnings.warn("quantum_electron.electron_counter has been renamed to quantum_electron.full_model; "
              "import FullModel from quantum_electron or quantum_electron.full_model instead.",
              DeprecationWarning, stacklevel=2)
