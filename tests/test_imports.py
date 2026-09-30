import importlib
import sys
import pytest


def test_electron_counter_alias():
    """The old module name quantum_electron.electron_counter still provides FullModel, with a DeprecationWarning."""
    from quantum_electron import FullModel
    sys.modules.pop("quantum_electron.electron_counter", None)
    with pytest.warns(DeprecationWarning, match="full_model"):
        module = importlib.import_module("quantum_electron.electron_counter")
    assert module.FullModel is FullModel
    assert FullModel.__module__ == "quantum_electron.full_model"
