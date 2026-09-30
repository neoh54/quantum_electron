import pytest
import numpy as np
from shapely import Polygon, Point
from quantum_electron import FullModel
from quantum_electron.initial_condition import InitialCondition
from quantum_electron.utils import r2xy

um = 1e-6


def make_ic() -> InitialCondition:
    x = np.linspace(-3, 3, 121)
    y = np.linspace(-3, 3, 121)
    X, Y = np.meshgrid(x, y)
    return InitialCondition({"dot": -(X ** 2 + Y ** 2).T, "xlist": x, "ylist": y}, {"dot": 1.0})


# L-shaped (non-convex) polygon in microns
L_shape = Polygon([(-2, -2), (2, -2), (2, -1), (-1, -1), (-1, 2), (-2, 2)])


@pytest.mark.parametrize("keep_off_boundary", [False, True])
def test_random_particles(keep_off_boundary):
    ic = make_ic()
    n, min_dist = 30, 0.3
    r = ic.random_particles(L_shape, n, min_dist, rng=1, keep_off_boundary=keep_off_boundary)

    assert r.shape == (2 * n,)
    x, y = r2xy(r)
    x, y = x / um, y / um

    # All electrons inside the polygon, and (optionally) at least min_dist/2 from its edge
    for xi, yi in zip(x, y):
        assert L_shape.contains(Point(xi, yi))
        if keep_off_boundary:
            assert L_shape.exterior.distance(Point(xi, yi)) >= min_dist / 2 - 1e-9

    # Pairwise spacing
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    assert d.min() >= min_dist

    # Reproducible with a seed
    assert np.array_equal(r, ic.random_particles(L_shape, n, min_dist, rng=1, keep_off_boundary=keep_off_boundary))


def test_random_particles_errors():
    ic = make_ic()
    small = Point(0, 0).buffer(0.1)

    with pytest.raises(ValueError, match="too small"):
        ic.random_particles(small, 2, 0.5, keep_off_boundary=True)

    with pytest.raises(RuntimeError, match="Placed only"):
        ic.random_particles(small, 50, 0.1, rng=0, max_tries=2000)


def test_random_particles_as_initial_condition():
    """The output can be passed directly to get_electron_positions."""
    ic = make_ic()
    r = ic.random_particles(Point(0, 0).buffer(1.0), 5, 0.2, rng=0)

    fm = FullModel(ic.potential_dict, ic.voltage_dict, trap_annealing_steps=[])
    res = fm.get_electron_positions(n_electrons=5, electron_initial_positions=r)
    assert res['success']
    assert np.all(np.abs(res['x']) < 1 * um)
