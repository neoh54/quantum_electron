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
    """The output can be passed directly to find_ground_configuration."""
    ic = make_ic()
    r = ic.random_particles(Point(0, 0).buffer(1.0), 5, 0.2, rng=0)

    fm = FullModel(ic.potential_dict, ic.voltage_dict, trap_annealing_steps=[])
    fm.find_ground_configuration(electron_initial_positions=r)
    assert fm.results.minimization_results['success']
    assert np.all(np.abs(fm.results.coordinates_final) < 1 * um)


def test_generate_initial_condition_stores_results():
    """generate_initial_condition places the electrons in the box (microns) and stores them (meters) in fm.results,
    from where find_ground_configuration starts."""
    fm = FullModel(make_ic().potential_dict, {"dot": 1.0}, trap_annealing_steps=[])
    box = Point(0.5, -0.5).buffer(0.4)
    assert fm.generate_initial_condition(6, box=box, min_dist=0.1, rng=0) is None

    assert fm.results.num_init == 6
    x, y = r2xy(fm.results.coordinates_init)
    assert all(box.contains(Point(xi / um, yi / um)) for xi, yi in zip(x, y))
    # Reproducible with a seed
    r0 = fm.results.coordinates_init.copy()
    fm.generate_initial_condition(6, box=box, min_dist=0.1, rng=0)
    assert np.array_equal(fm.results.coordinates_init, r0)

    fm.find_ground_configuration()
    assert fm.results.num_final == 6
    assert fm.results.minimization_results['success']
    assert np.array_equal(fm.results.coordinates_final, fm.results.minimization_results['x'])
