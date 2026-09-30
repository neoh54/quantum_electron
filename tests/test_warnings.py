import warnings
import pytest
import numpy as np
import scipy.optimize
from quantum_electron import FullModel, QuantumElectronWarning, ConvergenceWarning
from quantum_electron.initial_condition import InitialCondition

um = 1e-6


def make_well(**options) -> FullModel:
    x = np.linspace(-3, 3, 121)
    y = np.linspace(-3, 3, 121)
    X, Y = np.meshgrid(x, y)
    return FullModel({"dot": -(X ** 2 + Y ** 2).T, "xlist": x, "ylist": y}, {"dot": 1.0}, trap_annealing_steps=[], **options)


@pytest.fixture
def one_iteration(monkeypatch):
    """Limit scipy.optimize.minimize to a single iteration, such that the minimization does not converge (status > 0)."""
    minimize = scipy.optimize.minimize

    def limited(*args, **kwargs):
        kwargs['options'] = {**kwargs.get('options', {}), 'maxiter': 1}
        return minimize(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", limited)


@pytest.fixture
def no_progress(monkeypatch):
    """Replace scipy.optimize.minimize by a fake that returns the initial positions unchanged, flagged as not converged."""
    calls = []

    def fake_minimize(fun, x0, jac=None, **kwargs):
        x0 = np.asarray(x0, dtype=float)
        calls.append(len(x0) // 2)
        return scipy.optimize.OptimizeResult(x=x0, fun=fun(x0), jac=jac(x0), status=1, success=False, message="forced")

    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)
    return calls


def quantum_electron_warnings(record):
    return [w for w in record if issubclass(w.category, QuantumElectronWarning)]


def test_convergence_warning(one_iteration):
    fm = make_well()
    r0 = np.array([0.5, 0.5, -0.5, -0.5]) * um

    with pytest.warns(ConvergenceWarning, match="did not converge"):
        res = fm.get_electron_positions(n_electrons=2, electron_initial_positions=r0)
    assert res['status'] > 0

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fm.get_electron_positions(n_electrons=2, electron_initial_positions=r0, suppress_warnings=True)
    assert quantum_electron_warnings(record) == []


def test_warnings_can_be_turned_into_errors(one_iteration):
    fm = make_well()
    with warnings.catch_warnings():
        warnings.simplefilter("error", QuantumElectronWarning)
        with pytest.raises(ConvergenceWarning):
            fm.get_electron_positions(n_electrons=2, electron_initial_positions=np.array([0.5, 0.5, -0.5, -0.5]) * um)


def test_initial_condition_mismatch_warning():
    fm = make_well()
    with pytest.warns(QuantumElectronWarning, match="does not match n_electrons"):
        fm.get_electron_positions(n_electrons=3, electron_initial_positions=np.array([0.5, 0.5, -0.5, -0.5]) * um)


@pytest.mark.parametrize("suppress_warnings", [False, True])
def test_unbound_removal_independent_of_suppress_warnings(no_progress, suppress_warnings):
    """Removing unbound electrons (and restarting the minimization) must not depend on suppress_warnings."""
    fm = make_well(remove_unbound_electrons=True, remove_bounds=(-1 * um, 1 * um))
    r0 = np.array([0.5, 0.5, -0.5, -0.5, 2.5, 0.0]) * um  # the third electron is outside remove_bounds

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        res = fm.get_electron_positions(n_electrons=3, electron_initial_positions=r0, suppress_warnings=suppress_warnings)

    assert len(res['x']) == 4
    # The minimization is restarted with the 2 remaining electrons
    assert no_progress == [3, 2]
    messages = [str(w.message) for w in quantum_electron_warnings(record)]
    if suppress_warnings:
        assert messages == []
    else:
        assert any("1/3 unbound electrons removed" in m for m in messages)


def test_initial_condition_does_not_fit_warning():
    x = np.linspace(-3, 3, 121)
    y = np.linspace(-3, 3, 121)
    X, Y = np.meshgrid(x, y)
    ic = InitialCondition({"dot": -(X ** 2 + Y ** 2).T, "xlist": x, "ylist": y}, {"dot": 1.0})
    with pytest.warns(QuantumElectronWarning, match="Could not fit more than"):
        # The dot is a disk of radius 1 um, which cannot hold 500 electrons 0.5 um apart
        ic.make_by_chemical_potential(max_electrons=500, chemical_potential=1.0, min_spacing=0.5)


def test_invalid_resonator_mode():
    fm = make_well()
    fm.set_rf_interpolator(rf_electrode_labels=["dot"])
    resonator_dict = {'La': 20e-9, 'Lb': 20e-9, 'Ca': 50e-15, 'Cb': 50e-15, 'Cdot': 10e-15, 'mode': 'common'}
    with pytest.raises(ValueError, match="'common' was not understood"):
        fm.setup_eom_coupled_lc(np.array([0.0, 0.0]), resonator_dict=resonator_dict)
