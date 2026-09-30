import pytest
import numpy as np
from quantum_electron import FullModel
from scipy.constants import elementary_charge as qe, electron_mass as me


def make_model(**options) -> FullModel:
    """Two electrons in an isotropic parabolic confinement V(x, y) = (x^2 + y^2) / micron^2."""
    x = np.linspace(-3, 3, 401)
    y = np.linspace(-3, 3, 401)

    micron = 1e-6
    X, Y = np.meshgrid(x, y)
    potential_dict = {"dot": -((X * micron) ** 2 + (Y * micron) ** 2) / micron ** 2,
                      "xlist": x,
                      "ylist": y}

    fm = FullModel(potential_dict=potential_dict, voltage_dict={"dot": 1.0},
                   trap_annealing_steps=[], potential_smoothing=1e-7, **options)
    fm.set_rf_interpolator(rf_electrode_labels=["dot"])
    return fm


# Frequency of the center of mass mode in the parabolic trap: k = 2 e / micron^2
f_com = np.sqrt(2 * qe / 1e-12 / me) / (2 * np.pi)

# Coulomb, and Yukawa with a screening length much larger than the electron spacing, must give the same modes.
model_options = [{"include_screening": False},
                 {"include_screening": True, "screening_length": 1e3}]


@pytest.mark.parametrize("options", model_options)
def test_two_electron_modes(options):
    """For two electrons in a parabolic trap the in-plane modes are known analytically:
    two degenerate center of mass modes at f_com, the breathing mode at sqrt(3) f_com and the rotation mode at zero.
    """
    fm = make_model(**options)
    res = fm.get_electron_positions(n_electrons=2)

    K, M = fm.setup_eom(res['x'], resonator_dict=None)
    freqs, _ = fm.solve_eom(K, M, sort_by_cavity_participation=False)
    # The rotation mode has a zero eigenvalue, which can come out slightly negative (nan frequency)
    freqs = np.sort(np.nan_to_num(np.real(freqs)))

    assert freqs[0] / f_com < 1e-2
    assert freqs[1:] / f_com == pytest.approx([1, 1, np.sqrt(3)], rel=1e-3)


@pytest.mark.parametrize("options", model_options)
@pytest.mark.parametrize("Ltail", [0, 1e-9])
def test_two_electron_modes_coupled_lc(options, Ltail):
    """Same as above, but with the far-detuned (~5 GHz vs ~100 GHz) coupled LC resonator included."""
    fm = make_model(**options)
    res = fm.get_electron_positions(n_electrons=2)

    resonator_dict = {'La': 20e-9, 'Lb': 20e-9, 'Ca': 50e-15, 'Cb': 50e-15, 'Cdot': 10e-15, 'mode': 'diff', 'Ltail': Ltail}
    K, M = fm.setup_eom_coupled_lc(res['x'], resonator_dict=resonator_dict)
    freqs, _ = fm.solve_eom(K, M, sort_by_cavity_participation=False)
    freqs = np.sort(np.real(freqs))

    # Drop the rotation mode and the two resonator modes
    electron_freqs = freqs[freqs > 0.5 * f_com]
    assert electron_freqs / f_com == pytest.approx([1, 1, np.sqrt(3)], rel=1e-2)


@pytest.mark.parametrize("options", [{"include_screening": False},
                                     {"include_screening": True, "screening_length": 0.3e-6}])
def test_stiffness_matrix_equals_hessian(options):
    """The electron block of the EOM stiffness matrix must equal the Hessian of the total energy at equilibrium.
    A rotated, anisotropic trap makes sure the electron pairs are not aligned with the x or y axis, such that
    the xy cross terms (proportional to sin(2 theta)) are tested as well.
    """
    x = np.linspace(-3, 3, 401)
    y = np.linspace(-3, 3, 401)
    X, Y = np.meshgrid(x, y)
    potential_dict = {"dot": -(1.0 * X ** 2 + 2.5 * Y ** 2 + 1.2 * X * Y).T,
                      "xlist": x,
                      "ylist": y}

    fm = FullModel(potential_dict=potential_dict, voltage_dict={"dot": 1.0},
                   trap_annealing_steps=[], potential_smoothing=1e-9, **options)
    fm.set_rf_interpolator(rf_electrode_labels=["dot"])
    r = fm.get_electron_positions(n_electrons=3)['x']
    n = len(r) // 2

    # Finite difference Hessian of the total energy in J/m^2, reordered to match the EOM: [x0, ..., xn, y0, ..., yn]
    h = 1e-10
    H = np.zeros((2 * n, 2 * n))
    for i in range(2 * n):
        dr = np.zeros(2 * n)
        dr[i] = h
        H[:, i] = (fm.grad_total(r + dr) - fm.grad_total(r - dr)) / (2 * h) * qe
    order = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2))
    H = H[np.ix_(order, order)]

    K, _ = fm.setup_eom(r, resonator_dict=None)
    assert np.max(np.abs(K - H)) < 1e-4 * np.max(np.abs(H))

    resonator_dict = {'La': 20e-9, 'Lb': 20e-9, 'Ca': 50e-15, 'Cb': 50e-15, 'Cdot': 10e-15, 'mode': 'diff'}
    K, _ = fm.setup_eom_coupled_lc(r, resonator_dict=resonator_dict)
    assert np.max(np.abs(K[2:, 2:] - H)) < 1e-4 * np.max(np.abs(H))
