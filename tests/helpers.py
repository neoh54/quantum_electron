"""Shared helpers for the tests."""
import numpy as np
from quantum_electron.utils import xy2r, find_minimum_location


def ring_initial_condition(fm, n_electrons: int, radius_um: float = 0.18) -> np.ndarray:
    """Electrons on a ring of radius `radius_um` around the potential minimum: the deterministic initial condition that
    FullModel.get_electron_positions used by default before the stateful API.

    Returns:
        np.ndarray: [m] Electron positions [x0, y0, x1, y1, ...]
    """
    x0, y0 = find_minimum_location(fm.potential_dict, fm.voltage_dict)
    angles = 2 * np.pi * np.arange(n_electrons) / n_electrons
    return xy2r((x0 + radius_um * np.cos(angles)) * 1e-6, (y0 + radius_um * np.sin(angles)) * 1e-6)


def solve(fm, n_electrons: int = None, electron_initial_positions=None, **kwargs) -> dict:
    """Run find_ground_configuration (from a ring initial condition if none is given) and return the minimization result,
    i.e. what FullModel.get_electron_positions used to return."""
    if electron_initial_positions is None:
        electron_initial_positions = ring_initial_condition(fm, n_electrons)
    fm.find_ground_configuration(electron_initial_positions=electron_initial_positions, **kwargs)
    return fm.results.minimization_results
