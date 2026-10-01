import warnings
import pytest
import numpy as np
from quantum_electron import FullModel, Resonator
from helpers import solve

warnings.simplefilter("ignore", DeprecationWarning)


def make_model(n_electrons: int = 1) -> FullModel:
    """Anisotropic trap (f_x ~ 13.3 GHz, f_y ~ 9.4 GHz for one electron) and RF electrodes with uniform fields along x:
    'up' and 'down' (opposite fields, for the differential mode) and 'signal' (for a single-ended resonator)."""
    x = np.linspace(-3, 3, 241)
    X, Y = np.meshgrid(x, x)
    potential_dict = {"dot": -(X ** 2 + 0.5 * Y ** 2).T, "up": (0.05 * X).T, "down": (-0.05 * X).T, "signal": (0.1 * X).T,
                      "xlist": x, "ylist": x}
    fm = FullModel(potential_dict, {"dot": 0.02, "up": 0.0, "down": 0.0, "signal": 0.0}, trap_annealing_steps=[])
    solve(fm, n_electrons)
    fm.compute_spectrum()
    return fm


def exact_diff_shift(fm, L, C, Cdot):
    """Exact frequency shift of the differential mode from the coupled LC model (EOMSolver.setup_eom_coupled_lc)."""
    fm.set_rf_interpolator(["up", "down"])
    K, M = fm.setup_eom_coupled_lc(fm.results.coordinates_final, {'La': L, 'Lb': L, 'Ca': C, 'Cb': C, 'Cdot': Cdot, 'mode': 'diff'})
    f_all, _ = fm.solve_eom(K, M, sort_by_cavity_participation=False)
    f_all = np.real(f_all)
    f_bare = fm.f0_diff
    # nanargmin: zero modes (e.g. rotation of two electrons) can come out as nan
    return f_bare, f_all[np.nanargmin(np.abs(f_all - f_bare))] - f_bare


@pytest.mark.parametrize("n_electrons", [1, 2])
@pytest.mark.parametrize("Cdot", [1e-17, 1e-15])
def test_differential_shift_equals_coupled_lc(n_electrons, Cdot):
    """For a differential resonator (two nodes, Ca = Cb = C, coupled by Cdot), get_frequency_shift with C_d = C + 2 Cdot
    equals the exact shift of the coupled LC model, and it is negative because the electron modes are above f_r."""
    fm = make_model(n_electrons)
    L, C = 20e-9, 50e-15
    f_bare, df_exact = exact_diff_shift(fm, L, C, Cdot)

    resonator = Resonator(frequency=f_bare, capacitance=C + 2 * Cdot, mode='diff', electrodes=("up", "down"))
    df = fm.get_frequency_shift(resonator, gamma_e=0.0)

    assert df < 0
    assert df == pytest.approx(df_exact, rel=2e-3)


@pytest.mark.parametrize("f0", [5e9, 20e9])
def test_single_ended_shift_equals_lc(f0):
    """For a single-ended LC resonator, get_frequency_shift equals the exact shift from setup_eom. Below the electron modes
    (5 GHz) the shift is negative, above them (20 GHz) positive."""
    fm = make_model()
    fm.set_rf_interpolator(["signal"])
    Z0 = 50.0
    K, M = fm.setup_eom(fm.results.coordinates_final, {"f0": f0, "Z0": Z0})
    df_exact = fm.get_cavity_frequency_shift(K, M)
    fm.compute_spectrum()

    resonator = Resonator(frequency=f0, capacitance=1 / (2 * np.pi * f0 * Z0), mode='single', electrodes=("signal",))
    df = fm.get_frequency_shift(resonator, gamma_e=0.0)

    assert np.sign(df) == (-1 if f0 < 13e9 else 1)
    assert df == pytest.approx(df_exact, rel=2e-3)


def test_differential_coupling_has_sqrt2():
    """The differential mode couples sqrt(2) more strongly than a single-ended resonator with the same field and capacitance."""
    fm = make_model()
    fm.set_rf_interpolator(["up", "down"])
    g_diff = fm.get_coupling_to_mode(1, Resonator(5e9, 50e-15, 'diff'))
    g_single = fm.get_coupling_to_mode(1, Resonator(5e9, 50e-15, 'single'))
    assert g_diff == pytest.approx(np.sqrt(2) * g_single)
    # The y mode (index 0) does not couple to a field along x
    assert fm.get_coupling_to_mode(0, Resonator(5e9, 50e-15, 'diff')) == pytest.approx(0, abs=1e-6 * g_diff)


def test_susceptibility():
    fm = make_model(2)
    resonator = Resonator(5e9, 50e-15, 'diff', electrodes=("up", "down"))
    n_modes = len(fm.results.evals)

    # Real part gives the frequency shift; with damping, the imaginary part is nonzero (absorption)
    chi = fm.get_susceptibility(resonator, gamma_e=1e6)
    assert np.iscomplexobj(chi) and np.ndim(chi) == 0
    assert fm.get_frequency_shift(resonator, gamma_e=1e6) == pytest.approx(-5e9 * chi.real / 2)
    assert chi.imag < 0

    # Sum of the single-mode shifts
    total = sum(fm.get_frequency_shift_from_single_mode(n, resonator, 1e6) for n in range(n_modes))
    assert total == pytest.approx(fm.get_frequency_shift(resonator, 1e6))

    # Frequency response on an array of frequencies, and one damping per mode
    f = np.linspace(4e9, 6e9, 7)
    chi_f = fm.get_susceptibility(resonator, gamma_e=np.full(n_modes, 1e6), frequency=f)
    assert chi_f.shape == f.shape
    assert chi_f[3] == pytest.approx(chi)


def test_rf_field_from_resonator_electrodes():
    """The RF field is set from resonator.electrodes; without electrodes, set_rf_interpolator must have been called."""
    fm = make_model()
    with pytest.raises(ValueError, match="No RF field"):
        fm.get_coupling_to_mode(1, Resonator(5e9, 50e-15, 'diff'))

    g = fm.get_coupling_to_mode(1, Resonator(5e9, 50e-15, 'diff', electrodes=("up", "down")))
    assert fm.rf_electrode_labels == ("up", "down")
    fm.set_rf_interpolator(["up", "down"])
    assert fm.get_coupling_to_mode(1, Resonator(5e9, 50e-15, 'diff')) == pytest.approx(g)


def test_coupling_needs_electron_only_spectrum():
    fm = make_model()
    fm.set_rf_interpolator(["signal"])
    fm.compute_spectrum(resonator_dict={"f0": 5e9, "Z0": 50})
    with pytest.raises(ValueError, match="cavity coordinate"):
        fm.get_coupling_to_mode(1, Resonator(5e9, 50e-15, 'single'))


def test_invalid_resonator():
    with pytest.raises(ValueError, match="mode"):
        Resonator(5e9, 50e-15, 'common')
    with pytest.raises(ValueError, match="positive"):
        Resonator(-5e9, 50e-15, 'diff')
    with pytest.raises(ValueError, match="electrode"):
        Resonator(5e9, 50e-15, 'diff', electrodes=("up",))


def test_plot_electron_positions_invalid_state():
    fm = make_model()
    with pytest.raises(ValueError, match="'init' or 'final'"):
        fm.plot_electron_positions(state="fnal")
