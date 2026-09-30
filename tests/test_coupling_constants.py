import pytest
import numpy as np
from pathlib import Path
from quantum_electron import FullModel, PotentialVisualization, CouplingConstants, load_coupling_constants
from quantum_electron.initial_condition import InitialCondition
from quantum_electron.coupling_constants import read_ff_output, to_potential_dict
from quantum_electron.utils import make_potential

um = 1e-6
# Non-square grid on purpose: on a square grid, swapping x and y would not raise an error.
fem_file = Path(__file__).parent.parent / "examples" / "fem_data" / "nat_comm_dot_zoomed_in.txt"


def make_asymmetric():
    """Anisotropic well centered at (x0, y0) = (0.6, -0.3) um on a grid with nx != ny."""
    x = np.linspace(-2, 2, 81)
    y = np.linspace(-1, 1, 41)
    X, Y = np.meshgrid(x, y)  # shape (ny, nx), i.e. indexed [y, x]
    well = -((X - 0.6) ** 2 + 3 * (Y + 0.3) ** 2)
    potential_dict = {"dot": well.T, "xlist": x, "ylist": y}  # indexed [x, y]
    coupling_constants = CouplingConstants(x=x, y=y, data={"dot": well})  # indexed [y, x]
    return potential_dict, coupling_constants


def test_coupling_constants_equal_potential_dict():
    potential_dict, cc = make_asymmetric()
    voltages = {"dot": 1.0}

    fm_dict = FullModel(potential_dict, voltages, trap_annealing_steps=[])
    fm_cc = FullModel(cc, voltages, trap_annealing_steps=[])

    xs, ys = np.array([-1.0, 0.6, 1.5]) * um, np.array([0.5, -0.3, 0.0]) * um
    assert np.allclose(fm_cc.V(xs, ys), fm_dict.V(xs, ys))

    # A single electron must end up in the well at (0.6, -0.3) um, not at the swapped coordinates.
    res = fm_cc.get_electron_positions(n_electrons=1, electron_initial_positions=np.array([0.0, 0.0]))
    assert np.allclose(res['x'], [0.6 * um, -0.3 * um], atol=0.02 * um)

    for obj in [PotentialVisualization(cc, voltages), InitialCondition(cc, voltages)]:
        assert obj.potential_dict["dot"].shape == (len(cc.x), len(cc.y))


def test_load_coupling_constants_matches_raw_file():
    """load_coupling_constants must give the same model as passing the raw FreeFem dictionary (the old workflow)."""
    raw = read_ff_output(str(fem_file), '2Dmap')
    cc = load_coupling_constants(str(fem_file))

    assert len(cc.x) != len(cc.y)
    for electrode, array in cc.data.items():
        assert array.shape == (len(cc.y), len(cc.x))

    voltages = {"trap": 0.3, "resonator_1": 0.3, "resonator_2": 0.3, "gnd": -0.2, "trapgu": -0.1, "resgu": 0.0}
    assert np.allclose(make_potential(to_potential_dict(cc), voltages), make_potential(raw, voltages))


def test_orientation_agrees_with_zeroheliumkit():
    """zeroheliumkit's FieldAnalyzer computes the potential indexed as [y, x] from the same object."""
    fieldreader = pytest.importorskip("zeroheliumkit.fem.fieldreader")
    cc = load_coupling_constants(str(fem_file))
    zhk_cc = fieldreader.CouplingConstants(x=cc.x, y=cc.y, data=cc.data)

    voltages = {"trap": 0.3, "resonator_1": 0.3, "resonator_2": 0.3, "gnd": -0.2, "trapgu": -0.1, "resgu": 0.0}
    fa = fieldreader.FieldAnalyzer(zhk_cc)
    fa.set_voltages(voltages)

    fm = FullModel(zhk_cc, voltages)
    assert np.allclose(fa.potential, make_potential(fm.potential_dict, voltages).T)


def test_invalid_coupling_constants():
    _, cc = make_asymmetric()

    masked = CouplingConstants(x=cc.x, y=cc.y, data={"dot": np.ma.masked_less(cc.data["dot"], -3)})
    with pytest.raises(ValueError, match="masked"):
        to_potential_dict(masked)

    wrong_shape = CouplingConstants(x=cc.x, y=cc.y, data={"dot": cc.data["dot"].T})
    with pytest.raises(ValueError, match="shape"):
        to_potential_dict(wrong_shape)

    with pytest.raises(TypeError):
        to_potential_dict([cc.x, cc.y])
