class QuantumElectronWarning(UserWarning):
    """Base class for warnings issued by quantum_electron.

    Silence all of them with warnings.simplefilter("ignore", QuantumElectronWarning),
    or turn them into errors (e.g. in tests) with warnings.simplefilter("error", QuantumElectronWarning).
    """


class ConvergenceWarning(QuantumElectronWarning):
    """The minimization of the electron positions did not converge."""
