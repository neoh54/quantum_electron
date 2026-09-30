import pytest
import matplotlib
matplotlib.use('Agg')
import numpy as np
from quantum_electron import FullModel
from quantum_electron.utils import xy2r

um = 1e-6


def make_channel(**options) -> FullModel:
    """Long channel: x in (-20, 20) um, y in (-2, 2) um."""
    x = np.linspace(-20, 20, 201)
    y = np.linspace(-2, 2, 41)
    X, Y = np.meshgrid(x, y)
    potential_dict = {"dot": -(Y ** 2).T, "xlist": x, "ylist": y}
    return FullModel(potential_dict=potential_dict, voltage_dict={"dot": 1.0}, **options)


def test_default_remove_bounds_per_axis():
    """The default removal bounds are 95% of the simulation domain in x and in y separately."""
    fm = make_channel()
    assert np.allclose(fm.remove_bounds, ((-19 * um, 19 * um), (-1.9 * um, 1.9 * um)))

    x, y = fm._remove_unbound(xy2r(np.array([0, 0, 18, 19.5]) * um, np.array([0, 1.95, 0, 0]) * um))
    assert np.allclose(x, [0, 18 * um])
    assert np.allclose(y, [0, 0])


def test_legacy_remove_bounds():
    """A single (min, max) tuple applies to both x and y."""
    fm = make_channel(remove_bounds=(-1 * um, 1 * um))
    x, y = fm._remove_unbound(xy2r(np.array([0, 0, 1.5]) * um, np.array([0, 1.5, 0]) * um))
    assert np.allclose(x, [0])
    assert np.allclose(y, [0])


def test_dot_area_selects_contour_around_minimum():
    """Two wells: a deep wide one at x = -2 um (global minimum) and a shallow narrow one at x = +2 um.
    get_dot_area must return the area of the contour around the global minimum, not simply the last or largest contour."""
    x = np.linspace(-5, 5, 401)
    y = np.linspace(-5, 5, 401)
    X, Y = np.meshgrid(x, y)
    well = 1.0 * np.exp(-((X + 2) ** 2 + Y ** 2) / 1.0) + 0.5 * np.exp(-((X - 2) ** 2 + Y ** 2) / 0.2)
    fm = FullModel(potential_dict={"dot": well.T, "xlist": x, "ylist": y}, voltage_dict={"dot": 1.0})

    # The contour at -0.01 eV around the deep well is a circle with r^2 = ln(100)
    area = fm.get_dot_area(plot=True, barrier_location=(0, 4.5), barrier_offset=-0.01)
    assert area == pytest.approx(np.pi * np.log(100), rel=1e-2)
