![example workflow](https://github.com/gkoolstra/quantum_electron/actions/workflows/python-app.yml/badge.svg)
# Quantum Electron Solver
![image info](./images/electron_results.png)
## Main use cases
This package has two main functions
1. It simulates electron positions in a two dimensional plane for electrons confined in an electrostatic potential $\phi$. Electron-electron interactions are also taken into account. Physically, it minimizes the total energy of an $N$-electron system, which is given by $-e\sum_i \phi(\mathbf{r}_i) + \sum_{i<j} \frac{e^2}{4 \pi \epsilon_0} \frac{1}{|\mathbf{r}_i - \mathbf{r}_j|}$.

2. It calculates properties of in-plane electron modes. This is useful in two cases: (a) the electron motional states can be used for quantum computation, and this package can help to determine eigenfrequencies and eigenvectors of electron clusters (b) in the large $N$ limit the electron motional modes are also known as plasmons. There is an abundance of literature about these charge density waves, and many of the properties can be reproduced with this module.

### Features
- Robust operation through the use of the `scipy.optimize.minimize` library. We assure proper and fast convergence because the force (gradient of the energy) is supplied as an argument of the minimizer.
- Supply arbitrary potential energies $\phi$, as long as they're on a rectangular grid. 
- Handles problems up to $N \approx 400$ electrons in less than 1 minute on a laptop computer. 
- Periodic boundary condition support for systems with open boundaries.
- Seemless integration with finite element modeling software ZeroHeliumKit.

![Package performance](./images/performance.png)

## Installation

Clone this module in a directory of your choice

```
git clone https://github.com/gkoolstra/quantum_electron.git
```
In a terminal window, change into the cloned directory and install the package (Python 3.10 or newer) in editable mode, either with pip:
```
cd quantum_electron
pip install -e ".[test]"
```
or with [uv](https://docs.astral.sh/uv/), which creates a virtual environment in `.venv` and installs the package together with the development tools (pytest, ruff, nbstripout):
```
cd quantum_electron
uv sync
```
The source code lives in `src/quantum_electron`. To run the example notebooks, also install the `notebooks` extra (Jupyter, IPython for animations, pyvista for `select_outer_electrons`, alive_progress, sympy): `pip install -e ".[notebooks]"` or `uv sync --extra notebooks`. After installation, it is advised to take a look at the `examples` folder to explore some of the functionalities of this module. 

### Additional packages
To generate animations, this module relies on `ffmpeg`. On MacOS this can be easily installed using [homebrew](https://formulae.brew.sh/formula/ffmpeg) from the Terminal. On Windows it can be installed using the following [link](https://www.ffmpeg.org/download.html). 

This module also integrates well with the output of the FEM software [ZeroHeliumKit](https://github.com/eeroqlab/zeroheliumkit). Please refer to any dependencies for ZHK on the linked github page.

## Tests
To test the performance of the minimization, we're building and expanding a suite of tests based on the `pytest` framework. To run these tests, `cd` into the main module directory and run `pytest` (or `uv run pytest`). Currently, we have implemented a test in `test_wigner_molecules.py`, which compares the energy per particle of Wigner molecules in a parabolic confinement to known tabulated values.

## Getting started
The best way to learn how to use the module is to browse the examples. At a very high level this is the workflow:

To solve for the positions of the electrons, one can now use the following sets of short commands. Each step stores its result in `f.results`:
```
from shapely import Point
from quantum_electron import FullModel
f = FullModel(potential_dict, voltages, **options)
f.periodic_boundaries = ['x']

N = 58
# N electrons at random positions inside a disk of radius 1 micron, at least 0.05 micron apart
f.generate_initial_condition(N, box=Point(0, 0).buffer(1.0), min_dist=0.05, rng=0)
f.find_ground_configuration(verbose=False)

res = f.results.minimization_results          # output of scipy.optimize.minimize; f.results.coordinates_final [m]
f.plot_electron_positions("final")
```
A specific initial condition (e.g. the result of a previous step in a voltage sweep) can be passed directly: `f.find_ground_configuration(electron_initial_positions=r0)`.

The in-plane modes and their coupling to a resonator follow from the final configuration. A `Resonator` describes the resonator mode in a fixed format: its frequency, its capacitance, whether it is a differential (`'diff'`) or single-ended (`'single'`) mode, and its RF electrodes:
```
from quantum_electron import Resonator
resonator = Resonator(frequency=5e9, capacitance=50e-15, mode='diff', electrodes=('res_plus', 'res_min'))

f.compute_spectrum()                                     # f.results.evals [Hz] (ascending), f.results.evecs
g = f.get_coupling_to_mode(0, resonator)                 # [Hz]
chi = f.get_susceptibility(resonator, gamma_e=1e6)       # complex susceptibility at the resonator frequency
df = f.get_frequency_shift(resonator, gamma_e=1e6)       # [Hz], -f_r Re{chi} / 2
```
The coupling to mode n is $g_n = c\, e (\vec{E}\cdot\vec{x}_n) / (2\sqrt{m_e C})$ with $c=\sqrt{2}$ for the differential mode, and the susceptibility is $\chi_e(\omega) = \sum_n 4 g_n^2 / (\omega_n^2 - \omega^2 + 2 i \omega \Gamma_n)$, with $2\pi\Delta f \approx -\omega_r \mathrm{Re}\{\chi_e(\omega_r)/2\}$. For the differential mode, $C$ is the capacitance of each node to ground plus twice the capacitance between the nodes.

The first argument of `FullModel` can be a `potential_dict` (electrode names as keys with 2D arrays indexed as `[x, y]`, plus `'xlist'` and `'ylist'` in microns), or a `CouplingConstants` object from ZeroHeliumKit or from this package (attributes `x`, `y` and `data`, with arrays indexed as `[y, x]`). FreeFem `2Dmap` output files, such as those in `examples/fem_data`, can be loaded without ZeroHeliumKit:
```
from quantum_electron import FullModel, load_coupling_constants
couplings = load_coupling_constants("examples/fem_data/nat_comm_dot_zoomed_in.txt")
f = FullModel(couplings, voltages, **options)
```

There are a number of options that influence the solution of the minimization problem. Here is a dictionary of options that can be passed to `FullModel` to get started: 
```
options = {"include_screening" : True, # Include screening of electron-electron interactions due to thin film.
           "screening_length" : 2e-6, # Typically helium thickness.
           "potential_smoothing" : 5e-4, # Numerical smoothing of the splines (gets rid of some noise, can introduce artifacts)
           "remove_unbound_electrons" : False, # Removes electrons if they shot outside the solution box.
           "remove_bounds" : None, # Sets which electrons should be removed if above is True.
           "trap_annealing_steps" : [0.1] * 10, # List of annealing temperatures, length determines the number of steps
           "max_x_displacement" : 0.1e-6, # Maximum x-displacement of solved electron positions during annealing.
           "max_y_displacement" : 0.1e-6} # Maximum y-displacement of solved electron positions during annealing.
```

## Units
Lengths follow one rule:
- **Meters (SI)** for electron coordinates and everything that is compared to them: electron positions `r = [x0, y0, x1, y1, ...]` (including the initial condition, `f.results.coordinates_init` and `f.results.coordinates_final`), `remove_bounds`, `max_x_displacement` / `max_y_displacement`, the bounds of `count_electrons_in_dot`, and the `amplitude` of eigenvector animations. Energies are in eV, gradients in eV/m.
- **Microns** for everything that refers to the potential map or a plot window: `xlist` / `ylist` (and the `x`, `y` of a `CouplingConstants` object), `coor`, `dxdy`, `loc`, `center`, `barrier_location`, and the shapes and spacings passed to `InitialCondition` (`coor`, `dxdy`, `min_spacing`, `polygon`, `min_dist`).

In the docstrings, every length argument is tagged with its unit, `[m]` or `[microns]`. For example, `generate_initial_condition(n, box=Point(0, 0).buffer(0.2), min_dist=0.05)` places `n` electrons in a disk of radius 0.2 microns, at least 0.05 microns apart, and stores their positions in meters.

## Warnings
Problems during a calculation are reported as Python warnings rather than printed messages: a `ConvergenceWarning` if the minimization did not converge, and a `QuantumElectronWarning` (the base class) for e.g. removed or out-of-domain electrons. They can be filtered with the standard `warnings` module, for example in a voltage sweep:
```
import warnings
from quantum_electron import ConvergenceWarning, QuantumElectronWarning

warnings.simplefilter("ignore", QuantumElectronWarning)  # silence all quantum_electron warnings
warnings.simplefilter("error", ConvergenceWarning)       # or: raise an exception when a minimization does not converge
```
`find_ground_configuration(..., suppress_warnings=True)` silences them for a single call.

## Tips for the initial condition
The initial condition can affect the final minimization result quite strongly. We encourage you to take a look at the example notebook about initial conditions. If there are issues with convergence you can first check convergence with `f.plot_convergence()`. A good final value for the cost function is ~1-500 eV/m. If the lowest value of the cost function is signifantly higher than this, or if warnings appear, here are some rules of thumb for successful convergence:
1. Don't create an initial condition where too many electrons are placed in a small area.
2. Don't place electrons in an initial condition where the potential is too flat, such as on a ground plane. 
3. Be mindful of electron sinks, i.e. channels for electrons to escape. These can appear if an electrode is adjacent to the ground plane, and has an applied voltage that is more positive than the ground plane.

## Contributing
Contributions to this growing repository are welcome. Please feel free to create a fork and create a pull request with your suggested changes.

Notebook outputs are not stored in git. After cloning, register the [nbstripout](https://github.com/kynan/nbstripout) git filter once (it is included in the uv `dev` group, or `pip install nbstripout`):
```
nbstripout --install
```
The filter strips outputs and execution counts when notebooks are staged; your local copies keep their outputs.

By default `pytest` skips tests marked as slow (`tests/test_performance.py`). Run them with `pytest -m slow`, or everything with `pytest -m ""`.

## Credit
If you found this module useful in your research, please consider citing this code in your publication using a hyperlink.

## To-do list
- [ ] Standardize units of the arguments. The current convention is documented in the Units section, but a single unit for all arguments would be a breaking change (e.g. for a 1.0 release).
- [ ] Figure out how to handle warning messages for convergence issues. Why do problems sometimes have a hard time converging?
- [x] Split off the Schrodinger solver. It has been removed; this package now only does classical calculations.