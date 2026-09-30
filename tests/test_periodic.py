import pytest
import itertools
import numpy as np
from quantum_electron.position_solver import PositionSolver


@pytest.mark.parametrize("periodic_boundaries", [[], ['x'], ['y'], ['x', 'y']])
def test_minimum_image_distances(periodic_boundaries):
    """calculate_metrics must return the shortest distance between electrons, compared to a brute force search over
    the periodic images of the simulation domain."""
    x = np.linspace(-2e-6, 3e-6, 51)
    y = np.linspace(-1e-6, 1e-6, 21)
    ps = PositionSolver(x, y, np.zeros((len(x), len(y))))
    ps.periodic_boundaries = periodic_boundaries
    Lx, Ly = x[-1] - x[0], y[-1] - y[0]

    rng = np.random.default_rng(0)
    xi = rng.uniform(x[0], x[-1], 20)
    yi = rng.uniform(y[0], y[-1], 20)

    XiXj, YiYj, Rij = ps.calculate_metrics(xi, yi)

    nx = [-1, 0, 1] if 'x' in periodic_boundaries else [0]
    ny = [-1, 0, 1] if 'y' in periodic_boundaries else [0]
    for a, b in itertools.product(range(len(xi)), repeat=2):
        expected = min(np.hypot(xi[b] - xi[a] + kx * Lx, yi[b] - yi[a] + ky * Ly) for kx in nx for ky in ny)
        assert Rij[a, b] == pytest.approx(expected, rel=1e-12, abs=1e-18)
        assert np.hypot(XiXj[a, b], YiYj[a, b]) == pytest.approx(Rij[a, b], rel=1e-12, abs=1e-18)
