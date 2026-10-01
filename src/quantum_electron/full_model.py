"""FullModel: electron positions in an electrostatic potential, and their in-plane modes and coupling to a resonator.

Typical workflow (results are stored in FullModel.results, see Results):

    resonator = Resonator(frequency=5e9, capacitance=1e-12, mode='diff', electrodes=("res_plus", "res_min"))
    fm = FullModel(potential_dict, voltage_dict)
    fm.set_rf_interpolator(rf_electrode_labels=list(resonator.electrodes))
    fm.generate_initial_condition(n_electrons=5, box=Point(0, 0).buffer(0.5), min_dist=0.1)
    fm.find_ground_configuration()
    fm.compute_spectrum()
    df = fm.get_frequency_shift(resonator, gamma_e=1e6)
"""
import matplotlib
import matplotlib.animation as animation
import shapely.plotting
import numpy as np
import warnings

from typing import List, Dict, Optional, Union
from matplotlib import pyplot as plt
from matplotlib import patheffects as pe
from scipy.constants import elementary_charge as q_e, epsilon_0 as eps0, electron_mass as m_e
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.optimize import minimize
from shapely import Polygon, Point
from skimage import measure
from numpy.typing import ArrayLike
from dataclasses import dataclass, field


from .utils import find_nearest, xy2r, r2xy, find_minimum_location, make_potential
from .utils import PotentialVisualization
from .coupling_constants import CouplingConstants, to_potential_dict
from .position_solver import PositionSolver, ConvergenceMonitor
from .eom_solver import EOMSolver
from .initial_condition import InitialCondition
from .exceptions import QuantumElectronWarning, ConvergenceWarning
from .resonator import Resonator


@dataclass
class Results():
    """Results of the FullModel workflow, filled in step by step by the FullModel methods.

    Attributes:
        num_init (int): Number of electrons in the initial condition (generate_initial_condition).
        num_final (int): Number of electrons in the final configuration (find_ground_configuration). Also overwritten by
            count_electrons_in_dot with the number of electrons inside the given bounds.
        coordinates_init (ArrayLike): [m] Initial electron positions [x0, y0, x1, y1, ...] (generate_initial_condition).
        coordinates_final (ArrayLike): [m] Minimized electron positions [x0, y0, x1, y1, ...] (find_ground_configuration).
        evecs (ArrayLike): Eigenvectors of the in-plane modes as columns, sorted like evals (compute_spectrum). Without a resonator,
            the rows are ordered [x0, ..., xN, y0, ..., yN]; with a resonator, the first row is the cavity coordinate.
        evals (ArrayLike): [Hz] Mode frequencies in ascending order (compute_spectrum). Zero modes can come out as nan.
        minimization_results (dict): Output of scipy.optimize.minimize for the best solution (find_ground_configuration).
    """
    num_init: int = field(default_factory = int)
    num_final: int = field(default_factory = int)
    coordinates_init: list | np.ndarray = field(default_factory = list)
    coordinates_final: list | np.ndarray = field(default_factory = list)
    evecs: list | np.ndarray = field(default_factory = list)
    evals: list | np.ndarray = field(default_factory = list)
    minimization_results: dict = field(default_factory = dict)


class FullModel(EOMSolver, PositionSolver, PotentialVisualization):
    """Electrons in an electrostatic potential: equilibrium positions (PositionSolver), in-plane equations of motion and
    coupling to a resonator (EOMSolver), and plotting of the potential (PotentialVisualization). See the module docstring
    for the typical workflow; results are stored in self.results.
    """

    def __init__(
            self,
            potential_dict: Union[Dict[str, ArrayLike], CouplingConstants],
            voltage_dict: Dict[str, float],
            include_screening: bool = False,
            screening_length: float = np.inf,
            potential_smoothing: float = 5e-4,
            remove_unbound_electrons: bool = False,
            remove_bounds: Optional[tuple] = None,
            trap_annealing_steps: list = [0.1] * 5,
            max_x_displacement: float = 0.2e-6,
            max_y_displacement: float = 0.2e-6
            ) -> None:
        """This class can be used to determine the coordinates of electrons in an electrostatic potential and solve for the in-plane equations of motion.
        Typical usage:

        voltage_dict = {"trap" : 0.5, "res_plus" : 0.4, "res_min" : 0.4}
        fm = FullModel(potential_dict, voltage_dict)
        fm.set_rf_interpolator(rf_electrode_labels=["res_plus", "res_min"])
        fm.generate_initial_condition(n_electrons=5, box=Point(0, 0).buffer(0.5), min_dist=0.1)
        fm.find_ground_configuration()
        fm.compute_spectrum()

        Args:
            potential_dict (Union[Dict[str, ArrayLike], CouplingConstants]): Dictionary containing at least the keys also present in the voltages dictionary.
            The 2d-array associated with each key contains the coupling coefficient for the respective electrode in space.
            Alternatively, a CouplingConstants object (from quantum_electron or zeroheliumkit) with attributes x, y and data.
            voltage_dict (Dict[str, float]): Dictionary with electrode names as keys. The value associated with each key is the voltage
            applied to each electrode
            include_screening (bool, optional): Use a screened (Yukawa) electron-electron interaction instead of the Coulomb interaction.
            Defaults to False.
            screening_length (float, optional): [m] Screening length of the Yukawa interaction, typically twice the helium thickness.
            Defaults to np.inf (Coulomb).
            potential_smoothing (float, optional): Smoothing factor of the spline interpolation of the potential. Removes noise from FEM data,
            but can introduce artifacts. Defaults to 5e-4.
            remove_unbound_electrons (bool, optional): Remove electrons outside remove_bounds when the minimization does not converge, and restart.
            Defaults to False.
            remove_bounds (Optional[tuple]): [m] Electrons outside these bounds are removed if remove_unbound_electrons is True.
            Either ((xmin, xmax), (ymin, ymax)), or (min, max) which is applied to both x and y. Defaults to None, in which case 95% of the
            simulation domain is used in each direction.
            trap_annealing_steps (list, optional): [K] After the first minimization, the solution is perturbed len(trap_annealing_steps) times
            with thermal kicks at temperature trap_annealing_steps[0] and minimized again, keeping the lowest energy. An empty list disables
            annealing. Defaults to [0.1] * 5.
            max_x_displacement (float, optional): [m] Maximum thermal kick in the x-direction during annealing. Defaults to 0.2e-6.
            max_y_displacement (float, optional): [m] Maximum thermal kick in the y-direction during annealing. Defaults to 0.2e-6.
        """
        self.rf_interpolator = None
        self.rf_electrode_labels = ()

        potential_dict = to_potential_dict(potential_dict)
        self.potential_dict = potential_dict
        self.voltage_dict = voltage_dict

        self.include_screening = include_screening
        self.screening_length = screening_length
        self.potential_smoothing = potential_smoothing
        self.spline_order = 3
        if remove_bounds is None:
            # 95% of the simulation domain, in each direction separately
            bounds = []
            for key in ['xlist', 'ylist']:
                lo, hi = potential_dict[key][0] * 1e-6, potential_dict[key][-1] * 1e-6
                center, half_width = (lo + hi) / 2, 0.95 * (hi - lo) / 2
                bounds.append((center - half_width, center + half_width))
            self.remove_bounds = tuple(bounds)
        else:
            self.remove_bounds = remove_bounds
        self.remove_unbound_electrons = remove_unbound_electrons

        self.trap_annealing_steps = trap_annealing_steps
        self.max_x_displacement = max_x_displacement
        self.max_y_displacement = max_y_displacement

        potential = make_potential(potential_dict, voltage_dict)

        # Inherit methods from the PositionSolver, EOMSolver and PotentialVisualization classes
        PositionSolver.__init__(self, potential_dict['xlist'] * 1e-6, potential_dict['ylist'] * 1e-6, -potential,
                                spline_order_x=self.spline_order, spline_order_y=self.spline_order,
                                smoothing=self.potential_smoothing, include_screening=self.include_screening, screening_length=self.screening_length)

        EOMSolver.__init__(self, Ex=self.Ex, Ey=self.Ey,
                           Ex_up=self.Ex_up, Ex_down=self.Ex_down, Ey_up=self.Ey_up, Ey_down=self.Ey_down,
                           curv_xx=self.ddVdx, curv_xy=self.ddVdxdy, curv_yy=self.ddVdy)

        PotentialVisualization.__init__(
            self, potential_dict=potential_dict, voltages=voltage_dict)

        self.ConvergenceMonitor = ConvergenceMonitor

        self.results = Results()

    def set_rf_interpolator(self, rf_electrode_labels: List[str]) -> None:
        """Sets the rf_interpolator object, which allows evaluation of the electric field Ex and Ey at arbitrary coordinates. 
        This must be done before any calls to EOMSolver, such as setup_eom or solve_eom.

        The RF field Ex and Ey are determined from the same data as the DC fields, and are evaluated by setting +/- 0.5V on the 
        electrodes that couple to the RF-mode. These electrodes should be specified in the argument rf_electrode_labels.

        Args:
            rf_electrode_labels (List[str]): List of electrode names, these strings must also be present as keys in voltage_dict and potential_dict.
        """

        # Remember which electrodes define the RF field (see _set_rf_field)
        self.rf_electrode_labels = tuple(rf_electrode_labels)
        rf_electrode_labels = list(rf_electrode_labels)

        rf_voltage_dict = self.voltage_dict.copy()

        # For generating the rf electrodes, we must set all electrodes to 0.0 except the resoantor + and resonator - electrodes.
        for key in rf_voltage_dict.keys():
            rf_voltage_dict[key] = 0.0

        if len(rf_electrode_labels) == 2:
            rf_voltage_dict[rf_electrode_labels[0]] = +0.5
            rf_voltage_dict[rf_electrode_labels[1]] = -0.5
        elif len(rf_electrode_labels) == 1:
            rf_voltage_dict[rf_electrode_labels[0]] = +1.0
        else:
            raise ValueError(
                "More than 2 electrodes are not supported for the RF interpolator.")

        potential = make_potential(self.potential_dict, rf_voltage_dict)

        # By using the interpolator we create a function that can evaluate the potential energy for an electron at arbitrary x,y
        # This is useful if the original potential data is sparsely sampled (e.g. due to FEM time constraints)
        self.rf_interpolator = RectBivariateSpline(self.potential_dict['xlist']*1e-6,
                                                   self.potential_dict['ylist']*1e-6,
                                                   potential)

        # The code below is only for setting up the coupled LC circuit.
        # For the coupled LC circuit, we must consider the electric field generated by each electrode individually
        # In this case, rf_electrode_labels must contain at least 2 items
        if len(rf_electrode_labels) == 1:
            rf_electrode_labels *= 2

        assert len(rf_electrode_labels) == 2

        # We assume the first electrode is associated with the 'up' electrode
        rf_voltage_dict[rf_electrode_labels[0]] = 1.0
        rf_voltage_dict[rf_electrode_labels[1]] = 0.0

        potential = make_potential(self.potential_dict, rf_voltage_dict)

        # By using the interpolator we create a function that can evaluate the potential energy for an electron at arbitrary x,y
        # This is useful if the original potential data is sparsely sampled (e.g. due to FEM time constraints)
        self.rf_interpolator_up = RectBivariateSpline(self.potential_dict['xlist']*1e-6,
                                                      self.potential_dict['ylist']*1e-6,
                                                      potential)

        # Repeat for the 'down' electrode
        rf_voltage_dict[rf_electrode_labels[0]] = 0.0
        rf_voltage_dict[rf_electrode_labels[1]] = 1.0

        potential = make_potential(self.potential_dict, rf_voltage_dict)

        # By using the interpolator we create a function that can evaluate the potential energy for an electron at arbitrary x,y
        # This is useful if the original potential data is sparsely sampled (e.g. due to FEM time constraints)
        self.rf_interpolator_down = RectBivariateSpline(self.potential_dict['xlist']*1e-6,
                                                        self.potential_dict['ylist']*1e-6,
                                                        potential)

    def Ex_up(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the x-direction due to only the `up` electrode in the differential pair. 
        `setup_rf_interpolator` must be run prior to calling this function.
        This function is used by the setup_eom_coupled_lc function.

        Args:
            xe (ArrayLike): [m] array of x-coordinates where Ex should be evaluated.
            ye (ArrayLike): [m] array of y-coordinates where Ex should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator_up.ev(xe, ye, dx=1)

    def Ex_down(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the x-direction due to only the `down` electrode in the differential pair. 
        `setup_rf_interpolator` must be run prior to calling this function.
        This function is used by the setup_eom_coupled_lc function.

        Args:
            xe (ArrayLike): [m] array of x-coordinates where Ex should be evaluated.
            ye (ArrayLike): [m] array of y-coordinates where Ex should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator_down.ev(xe, ye, dx=1)

    def Ey_up(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the y-direction due to only the `up` electrode in the differential pair. 
        `setup_rf_interpolator` must be run prior to calling this function.
        This function is used by the setup_eom_coupled_lc function.

        Args:
            xe (ArrayLike): [m] array of x-coordinates where Ey should be evaluated.
            ye (ArrayLike): [m] array of y-coordinates where Ey should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator_up.ev(xe, ye, dy=1)

    def Ey_down(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the y-direction due to only the `down` electrode in the differential pair. 
        `setup_rf_interpolator` must be run prior to calling this function.
        This function is used by the setup_eom_coupled_lc function.

        Args:
            xe (ArrayLike): [m] array of x-coordinates where Ey should be evaluated.
            ye (ArrayLike): [m] array of y-coordinates where Ey should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator_down.ev(xe, ye, dy=1)

    def Ex(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the x-direction from the rf_interpolator. setup_rf_interpolator must be run prior to calling this function.
        This function is used by the setup_eom function.

        Args:
            xe (ArrayLike): [m] array of x-coordinates where Ex should be evaluated.
            ye (ArrayLike): [m] array of y-coordinates where Ex should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator.ev(xe, ye, dx=1)

    def Ey(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the y-direction from the rf_interpolator. setup_rf_interpolator must be run prior to calling this function.
        This function is used by the setup_eom function.

        Args:
            xe (ArrayLike): [m] array of x-coordinates where Ey should be evaluated.
            ye (ArrayLike): [m] array of y-coordinates where Ey should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator.ev(xe, ye, dy=1)


    def generate_initial_condition(self, n_electrons: int, box: Polygon, min_dist: float, **kwargs) -> ArrayLike:
        """Generates a random initial condition: n_electrons placed at random inside `box`, at least `min_dist` apart
        (see InitialCondition.random_particles). The result is stored in self.results.coordinates_init (in meters) and
        self.results.num_init, and is the default starting point of find_ground_configuration.

        Args:
            n_electrons (int): Number of electrons.
            box (Polygon): [microns] shapely Polygon of the area in which the electrons are placed, e.g. Point(0, 0).buffer(0.5).
            min_dist (float): [microns] Minimum spacing between electrons.
            **kwargs: Passed to InitialCondition.random_particles: rng (seed or numpy Generator), max_tries, keep_off_boundary.

        Raises:
            ValueError: If the polygon is too small for the requested min_dist.
            RuntimeError: If not all electrons could be placed.

        Returns:
            None: the initial condition is stored in self.results.
        """
        ic = InitialCondition(self.potential_dict, self.voltage_dict)
        self.results.num_init = n_electrons
        self.results.coordinates_init = ic.random_particles(box, n_electrons, min_dist, **kwargs)


    def count_electrons_in_dot(self, r: ArrayLike, trap_bounds_x: tuple = (-1e-6, 1e-6), trap_bounds_y: tuple = (-1e-6, 1e-6)) -> float:
        """Counts the number of coordinate pairs in r that fall within the bounds specified by trap_bounds_x and trap_bounds_y

        Args:
            r (ArrayLike): [m] Electron coordinates of length 2 * n_electrons. This should be in the order [x0, y0, x1, y1, ...]
            trap_bounds_x (tuple, optional): [m] Electrons will be counted if they fall within this x-domain. Defaults to (-1e-6, 1e-6).
            trap_bounds_y (tuple, optional): [m] Electrons will be counted if they fall within this y-domain. Defaults to (-1e-6, 1e-6).

        Returns:
            int: Number of electrons within the confines of the dot. This number is also stored in self.results.num_final.
        """
        ex, ey = r2xy(r)
        x_ok = np.logical_and(ex < trap_bounds_x[1], ex > trap_bounds_x[0])
        y_ok = np.logical_and(ey < trap_bounds_y[1], ey > trap_bounds_y[0])
        x_and_y_ok = np.logical_and(x_ok, y_ok)
        self.results.num_final = np.sum(x_and_y_ok)
        return np.sum(x_and_y_ok)

    def get_dot_area(self, plot: bool = True, barrier_location: tuple = (-1, 0), barrier_offset: float = -0.01, **kwargs) -> float:
        """Finds the area of the dot spanned by the points that lie on a equipotential that is determined by the 
        `barrier_location` and `barrier_offset`. The resulting area has the same units as self.potential_dict['xlist'] ** 2

        Args:
            plot (bool, optional): Plot the contour and polygon spanned by that contour. Defaults to True.
            barrier_location (tuple, optional): [microns] Location (x, y) in the map where to measure the barrier_height. The contour
            will be drawn `barrier_offset` above the potential value at the barrier_location. Defaults to (-1, 0).
            barrier_offset (float, optional): barrier_offset in eV. The contour will be drawn with this offset. Defaults to -0.01 (eV).

        Returns:
            float: Area
        """
        potential = make_potential(self.potential_dict, self.voltage_dict)

        idx = find_nearest(self.potential_dict['ylist'], barrier_location[1])
        idy = find_nearest(self.potential_dict['xlist'], barrier_location[0])
        barrier_height = -potential[idy, idx]

        # Contour can return non-integer indices (it interpolates to find the contour)
        # Thus we need to create a mappable for x and y.
        fx = interp1d(
            np.arange(len(self.potential_dict['xlist'])), self.potential_dict['xlist'])
        fy = interp1d(
            np.arange(len(self.potential_dict['ylist'])), self.potential_dict['ylist'])

        # Use sci-kit image function measure to find the contours.
        contours = measure.find_contours(-potential.T,
                                         barrier_height + barrier_offset)

        # A contour needs at least 3 points to span an area
        polygons = [Polygon(np.c_[fx(contour[:, 1]), fy(contour[:, 0])]) for contour in contours if len(contour) >= 3]

        if len(polygons) > 0:
            # There may be multiple contours: pick the one that encloses the potential minimum, or else the largest one.
            minimum = Point(find_minimum_location(self.potential_dict, self.voltage_dict))
            enclosing = [p for p in polygons if p.contains(minimum)]
            p = min(enclosing, key=lambda p: p.area) if enclosing else max(polygons, key=lambda p: p.area)

            if plot:
                shapely.plotting.plot_polygon(p, **kwargs)
                plt.grid(None)

            return p.area
        else:
            # If there are no contours, the situation is easy
            return 0.0

    def _remove_unbound(self, r: ArrayLike) -> tuple:
        """Removes electrons that lie outside self.remove_bounds.

        Args:
            r (ArrayLike): Electron coordinates in the order [x0, y0, x1, y1, ...]

        Returns:
            tuple: x and y coordinates of the electrons that remain.
        """
        # Support the legacy format (min, max), which applies to both x and y.
        if np.ndim(self.remove_bounds) == 1:
            xbounds = ybounds = self.remove_bounds
        else:
            xbounds, ybounds = self.remove_bounds

        x, y = r2xy(r)
        outside = (x < xbounds[0]) | (x > xbounds[1]) | (y < ybounds[0]) | (y > ybounds[1])
        return x[~outside], y[~outside]

    def find_ground_configuration(
            self,
            electron_initial_positions: Optional[ArrayLike] = None,
            verbose: bool = False,
            suppress_warnings: bool = False
            ) -> dict:
        """This is the main method to calculate the electron positions in an electrostatic potential. This function can be called with a specific initial condition, 
        which can be useful during voltage sweeps, or with the initial condition from generate_initial_condition (self.results.coordinates_init).

        The result is stored in self.results: coordinates_final [m], num_final and minimization_results. Upon running this function,
        useful feedback about the convergence can be found in the attribute CM.

        Args:
            electron_initial_positions (Optional[ArrayLike], optional): [m] Electron initial positions in the form [x0, y0, x1, y1, ...].
            Defaults to None, in which case self.results.coordinates_init is used.
            verbose (bool, optional): Prints convergence information. Defaults to False.
            suppress_warnings (bool, optional): If True, no QuantumElectronWarning (e.g. ConvergenceWarning) is issued. This is equivalent to
            warnings.simplefilter("ignore", QuantumElectronWarning), and does not change the result. Defaults to False.

        Returns:
            None: the result is stored in self.results. self.results.minimization_results is the dictionary returned from
            scipy.optimize.minimize. Some useful attributes in this dictionary: 'status' > 0 means the minimization failed.
            'x' contains the best solution [m] in the form [x0, y0, x1, y1, ...], which minimizes the gradient contained in 'jac' [eV/m].
            'fun' is the total energy [eV].
        """

        if electron_initial_positions is None:
            electron_initial_positions = self.results.coordinates_init
        if len(electron_initial_positions) == 0:
            raise ValueError("No initial condition: call generate_initial_condition first, or pass electron_initial_positions.")

        self.CM = self.ConvergenceMonitor(
            self.Vtotal, self.grad_total, call_every=1, verbose=verbose)

        # Convergence can happen one of two ways
        # (a) if the gradient self.grad_total(res['x']) < gradient_tolerance
        # (b) if res['fun'] changes less than the floating point precision from one iteration to the next.
        gradient_tolerance = 1e-1  # Units are eV/m

        # For improved performance we use maxls=100. Default is 20, but if starting close to the final solution, sometimes more
        # line searches are needed to converge. This is also helpful if the function landscape is very flat.
        trap_minimizer_options = {'method': 'L-BFGS-B',
                                  'jac': self.grad_total,
                                  'options': {'disp': False, 'gtol': gradient_tolerance, 'maxls': 100},
                                  'callback': self.CM.monitor_convergence}

        # initial_jacobian = self.grad_total(electron_initial_positions)
        res = minimize(self.Vtotal, electron_initial_positions, **trap_minimizer_options)

        no_electrons_left = False
        while res['status'] > 0:

            # Try removing unbounded electrons and restart the minimization
            if self.remove_unbound_electrons:
                # Remove any electrons that are outside self.remove_bounds
                best_x, best_y = self._remove_unbound(res['x'])

                # Use the solution from the current time step as the initial condition for the next timestep!
                electron_initial_positions = xy2r(best_x, best_y)
                if len(best_x) == len(res['x'][::2]):
                    # No electrons were removed: the simulation didn't converge for other reasons...
                    break

                if not suppress_warnings:
                    warnings.warn(f"{len(res['x'][::2]) - len(best_x)}/{len(res['x'][::2])} unbound electrons removed. "
                                  f"{len(best_x)} electrons remain.", QuantumElectronWarning, stacklevel=2)

                if len(electron_initial_positions) > 0:
                    if verbose:
                        print("Restart minimization!")
                    self.CM = self.ConvergenceMonitor(
                        self.Vtotal, self.grad_total, call_every=1, verbose=verbose)
                    trap_minimizer_options['callback'] = self.CM.monitor_convergence
                    res = minimize(self.Vtotal, electron_initial_positions, **trap_minimizer_options)
                else:
                    no_electrons_left = True
                    break
            else:
                best_x, best_y = r2xy(res['x'])
                idxs = np.where((best_x < self.x_min) | (best_x > self.x_max) |
                                (best_y < self.y_min) | (best_y > self.y_max))[0]
                if len(idxs) > 0 and (not suppress_warnings):
                    coordinates = ", ".join(f"({best_x[i] * 1E6:.3f}, {best_y[i] * 1E6:.3f})" for i in idxs)
                    warnings.warn(f"{len(idxs)} electrons are outside the simulation domain, at (x, y) = {coordinates} um.",
                                  QuantumElectronWarning, stacklevel=2)
                # To skip the infinite while loop.
                break

        if res['status'] > 0 and not (no_electrons_left) and not (suppress_warnings):
            warnings.warn(f"Initial minimization did not converge ({res['message']}). "
                          f"Final L-inf norm of gradient = {np.max(np.abs(res['jac'])):.2f} eV/m. "
                          "Please check your initial condition, are all electrons confined in the simulation area?",
                          ConvergenceWarning, stacklevel=2)

        if len(self.trap_annealing_steps) > 0:
            if verbose and res['status'] == 0:
                print("SUCCESS: Initial minimization for Trap converged!")
                # This maps the electron positions within the simulation domain
                print("Perturbing solution %d times at %.2f K. (dx,dy) ~ (%.3f, %.3f) µm..."
                      % (len(self.trap_annealing_steps), self.trap_annealing_steps[0],
                          np.mean(self.thermal_kick_x(res['x'][::2], res['x'][1::2], self.trap_annealing_steps[0],
                                                      maximum_dx=self.max_x_displacement)) * 1E6,
                          np.mean(self.thermal_kick_y(res['x'][::2], res['x'][1::2], self.trap_annealing_steps[0],
                                                      maximum_dy=self.max_y_displacement)) * 1E6))

            best_res = self.perturb_and_solve(self.Vtotal, len(self.trap_annealing_steps), self.trap_annealing_steps[0],
                                              res, maximum_dx=self.max_x_displacement, maximum_dy=self.max_y_displacement,
                                              do_print=verbose,
                                              **trap_minimizer_options)
        else:
            best_res = res

        if self.remove_unbound_electrons:
            best_x, best_y = self._remove_unbound(best_res['x'])
            best_res['x'] = xy2r(best_x, best_y)

        self.results.minimization_results = best_res
        self.results.coordinates_final = best_res["x"]
        self.results.num_final = len(best_res["x"]) // 2


    def compute_spectrum(self, resonator_dict: dict=None):
        """Computes the in-plane modes of the electrons at self.results.coordinates_final (see setup_eom and solve_eom).
        The frequencies are stored in self.results.evals [Hz] in ascending order, and the eigenvectors in self.results.evecs
        (columns, in the same order). set_rf_interpolator must be called first.

        Args:
            resonator_dict (dict, optional): Resonator parameters for setup_eom, with keys 'f0' [Hz] and 'Z0' [Ohm]. Defaults to None,
            in which case only the electron modes are computed. get_coupling_to_mode and get_frequency_shift require resonator_dict=None.
        """
        K, M = self.setup_eom(self.results.coordinates_final, resonator_dict)
        eigenfrequencies, evecs = self.solve_eom(K, M)

        # sort out
        ind = np.argsort(eigenfrequencies)
        self.results.evals = eigenfrequencies[ind]
        self.results.evecs = evecs[:, ind]


    def _set_rf_field(self, resonator: Resonator) -> None:
        """Sets the RF field from the electrodes of the resonator, unless it was already set for these electrodes.

        Args:
            resonator (Resonator): Resonator, see quantum_electron.Resonator.

        Raises:
            ValueError: If the resonator has no electrodes and set_rf_interpolator was not called.
        """
        if resonator.electrodes:
            if resonator.electrodes != self.rf_electrode_labels:
                self.set_rf_interpolator(list(resonator.electrodes))
        elif self.rf_interpolator is None:
            raise ValueError("No RF field: specify the electrodes of the resonator, or call set_rf_interpolator first.")

    def _electron_mode_frequencies(self, mode_id: Optional[int] = None) -> ArrayLike:
        """Frequencies of the electron modes in self.results.evals, with nan (imaginary frequencies: zero modes, or an unstable
        configuration) replaced by 0 Hz and a QuantumElectronWarning.

        Args:
            mode_id (Optional[int], optional): Index of a single mode. Defaults to None (all modes).

        Returns:
            ArrayLike: [Hz] Mode frequencies (or the frequency of mode `mode_id`).
        """
        f_e = np.real(np.asarray(self.results.evals, dtype=complex))
        if mode_id is not None:
            f_e = f_e[mode_id]
        n_nan = np.sum(~np.isfinite(f_e))
        if n_nan > 0:
            warnings.warn(f"{n_nan} electron mode(s) have an imaginary frequency (zero modes, or an unstable configuration). "
                          "They are treated as 0 Hz.", QuantumElectronWarning, stacklevel=3)
        return np.nan_to_num(f_e, nan=0.0, posinf=0.0, neginf=0.0)

    def get_coupling_to_mode(self, mode_id: int, resonator: Resonator) -> float:
        """Coupling strength between electron mode `mode_id` and the resonator,
        g_n = c e (E . x_n) / (2 sqrt(m_e C)) / (2 pi), where E is the RF field per volt at the electron positions (along x and y),
        x_n the normalized eigenvector, C = resonator.capacitance, and c = sqrt(2) for the differential mode (resonator.mode='diff')
        or 1 for a single-ended resonator.

        Requires compute_spectrum() (without resonator_dict, such that the eigenvectors contain only electron coordinates).
        The RF field is set from resonator.electrodes, or else must be set with set_rf_interpolator.

        Args:
            mode_id (int): Index of the mode in self.results.evals (ascending frequency).
            resonator (Resonator): Resonator, see quantum_electron.Resonator.

        Returns:
            float: [Hz] Coupling strength g_n / (2 pi).
        """
        if np.shape(self.results.evecs)[0] != len(self.results.coordinates_final):
            raise ValueError("The eigenvectors contain the cavity coordinate. Call compute_spectrum() without resonator_dict.")
        self._set_rf_field(resonator)

        eigen_vector = self.results.evecs[:, mode_id]
        eigen_vector_norm = eigen_vector / np.linalg.norm(eigen_vector)
        xlist, ylist = r2xy(self.results.coordinates_final)

        # RF field (per volt) at the electron locations, ordered as the eigenvector: [x0, ..., xN, y0, ..., yN]
        field_vector = np.concatenate([self.rf_interpolator.ev(xlist, ylist, dx=1),
                                       self.rf_interpolator.ev(xlist, ylist, dy=1)])

        coupling_strength = resonator.mode_factor * q_e * np.dot(eigen_vector_norm, field_vector) \
            / (2 * np.sqrt(m_e * resonator.capacitance)) / (2 * np.pi)

        return np.abs(coupling_strength)

    def get_susceptibility(self, resonator: Resonator, gamma_e: Union[float, ArrayLike],
                           frequency: Optional[Union[float, ArrayLike]] = None) -> Union[complex, ArrayLike]:
        """Complex electron susceptibility seen by the resonator,
        chi_e(f) = sum_n 4 g_n^2 / (f_n^2 - f^2 + 2 i f gamma_n), with g_n from get_coupling_to_mode and f_n = self.results.evals.
        This is the same as chi_e(omega) = sum_n 4 g_n^2 / (omega_n^2 - omega^2 + 2 i omega Gamma_n) with all frequencies in rad/s
        and Gamma_n = 2 pi gamma_n. The real part gives the frequency shift (get_frequency_shift), the imaginary part the absorption
        by the electrons.

        Args:
            resonator (Resonator): Resonator, see quantum_electron.Resonator.
            gamma_e (Union[float, ArrayLike]): [Hz] Damping gamma_n = Gamma_n / (2 pi) of the electron modes; a single value for all modes,
            or one value per mode in self.results.evals.
            frequency (Optional[Union[float, ArrayLike]], optional): [Hz] Frequency (or array of frequencies) f at which chi_e is
            evaluated. Defaults to None, in which case the resonator frequency is used.

        Returns:
            Union[complex, ArrayLike]: Dimensionless complex susceptibility chi_e(f).
        """
        f = resonator.frequency if frequency is None else np.asarray(frequency, dtype=float)
        f_e = self._electron_mode_frequencies()
        gamma = np.broadcast_to(np.asarray(gamma_e, dtype=float), f_e.shape)
        g = np.array([self.get_coupling_to_mode(n, resonator) for n in range(len(f_e))])

        f_grid = np.asarray(f)[..., None]
        chi = np.sum(4 * g ** 2 / (f_e ** 2 - f_grid ** 2 + 2j * f_grid * gamma), axis=-1)
        return chi[()] if np.ndim(f) == 0 else chi

    def get_frequency_shift_from_single_mode(self, mode_id: int, resonator: Resonator, gamma_e: float) -> float:
        """Resonator frequency shift due to a single electron mode, delta_f_n = -f_r Re{chi_n(f_r)} / 2, with
        chi_n(f_r) = 4 g_n^2 / (f_n^2 - f_r^2 + 2 i f_r gamma_e) (see get_susceptibility). The shift is negative for f_n > f_r.

        Args:
            mode_id (int): Index of the mode in self.results.evals (ascending frequency).
            resonator (Resonator): Resonator, see quantum_electron.Resonator.
            gamma_e (float): [Hz] Damping gamma_n = Gamma_n / (2 pi) of the electron mode.

        Returns:
            float: [Hz] Frequency shift of the resonator.
        """
        f_e = self._electron_mode_frequencies(mode_id)
        f_r = resonator.frequency
        g = self.get_coupling_to_mode(mode_id, resonator)
        chi = 4 * g ** 2 / (f_e ** 2 - f_r ** 2 + 2j * f_r * gamma_e)
        return -f_r * np.real(chi) / 2

    def get_frequency_shift(self, resonator: Resonator, gamma_e: Union[float, ArrayLike]) -> float:
        """Total resonator frequency shift due to all electron modes, delta_f = -f_r Re{chi_e(f_r)} / 2 (see get_susceptibility).

        Args:
            resonator (Resonator): Resonator, see quantum_electron.Resonator.
            gamma_e (Union[float, ArrayLike]): [Hz] Damping gamma_n = Gamma_n / (2 pi) of the electron modes; a single value for all
            modes, or one value per mode in self.results.evals.

        Returns:
            float: [Hz] Frequency shift of the resonator.
        """
        return -resonator.frequency * np.real(self.get_susceptibility(resonator, gamma_e)) / 2

    def plot_electron_positions(self, state: str="init", ax=None, color: str = 'mediumseagreen', marker_size: float = 10.0, shadow: bool=True, **kwargs) -> None:
        """Plot the initial or final electron positions stored in self.results.

        Args:
            state (str, optional): 'init' for self.results.coordinates_init (generate_initial_condition), or 'final' for
            self.results.coordinates_final (find_ground_configuration). Defaults to "init".
            ax (optional): Matplotlib axes object. Defaults to None, in which case the current axes are used.
            color (str, optional): Color of the markers representing the electrons. Defaults to 'mediumseagreen'.
            marker_size (float, optional): Marker size. Defaults to 10.0.
            shadow (bool, optional): Draw a shadow below the markers. Defaults to True.
            **kwargs: Passed to matplotlib's plot.
        """
        if state not in ["init", "final"]:
            raise ValueError(f"state = {state!r} was not understood. Please specify either 'init' or 'final'.")

        match state:
            case "init":
                x, y = r2xy(self.results.coordinates_init)
            case "final":
                x, y = r2xy(self.results.coordinates_final)

        if ax is None:
            if shadow:
                plt.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size,
                        path_effects=[pe.SimplePatchShadow(), pe.Normal()], **kwargs)
            else:
                plt.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size, **kwargs)
        else:
            if shadow:
                ax.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size,
                        path_effects=[pe.SimplePatchShadow(), pe.Normal()], **kwargs)
            else:
                ax.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size, **kwargs)


    def plot_eigenvector(self, mode_id: int, ax=None, length = 0.5, **kwargs):
        """Plot the final electron positions with the eigenvector of mode `mode_id` as arrows (see EOMSolver.plot_eigenvector).
        Overrides EOMSolver.plot_eigenvector, which takes the electron positions and eigenvector as arguments.

        Args:
            mode_id (int): Index of the mode in self.results.evals (ascending frequency).
            ax (optional): Matplotlib axes object. Defaults to None, in which case the current axes are used.
            length (float, optional): [microns] Length of the arrows. Defaults to 0.5.
            **kwargs: Passed to plot_electron_positions.
        """
        self.plot_electron_positions(state="final", ax=ax, **kwargs)
        super().plot_eigenvector(self.results.coordinates_final, self.results.evecs[:,mode_id], ax, length, "white")



    def animate_voltage_sweep(self, fig, ax, list_of_voltages: list, list_of_electron_positions: list, coor: tuple = (0, 0), dxdy: tuple = (2, 2), 
                              frame_interval_ms: int = 10, print_voltages: bool = False) -> matplotlib.animation.FuncAnimation:
        """
        Animates a voltage sweep by updating the voltage and electron positions over time. 
        This function only animates the sweep, it does not calculate the electron positions. This needs to be done beforehand.

        Args:
            list_of_voltages (list): A list of dictionaries representing the voltages at each frame.
            list_of_electron_positions (list): [m] A list of arrays representing the electron positions at each frame.
            coor (tuple, optional): [microns] The coordinates of the center of the plot. Defaults to (0, 0).
            dxdy (tuple, optional): [microns] The width and height of the plot. Defaults to (2, 2).
            frame_interval_ms (int, optional): The time interval between frames in milliseconds. Defaults to 10.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.

        Raises:
            AssertionError: If the length of the voltage list is not the same as the list of electron positions.
        """
        assert len(list_of_voltages) == len(
            list_of_electron_positions), "The length of the voltage list must be the same as the list of electron positions."

        potential = make_potential(self.potential_dict, list_of_voltages[0])
        zdata = -potential.T

        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        img_data = ax.imshow(zdata[::-1, :], cmap=plt.cm.RdYlBu_r, extent=[np.min(self.potential_dict['xlist']), np.max(self.potential_dict['xlist']),
                                                                           np.min(self.potential_dict['ylist']), np.max(self.potential_dict['ylist'])])

        final_x, final_y = r2xy(list_of_electron_positions[0])
        pts_data = ax.plot(final_x*1e6, final_y*1e6, 'ok', mfc='mediumseagreen', mew=0.5, ms=10,
                           path_effects=[pe.SimplePatchShadow(), pe.Normal()])

        cbar = plt.colorbar(img_data, fraction=0.046, pad=0.04)
        tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.set_ylabel(r"Potential energy $-eV(x,y)$")

        xmin, xmax = (coor[0] - dxdy[0]/2, coor[0] + dxdy[0]/2)
        ymin, ymax = (coor[1] - dxdy[1]/2, coor[1] + dxdy[1]/2)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        text_boxes = list()
        initial_voltages = list_of_voltages[0]
        if print_voltages:
            for k, electrode in enumerate(initial_voltages.keys()):
                text_boxes.append(ax.text(xmin - 0.75,
                                        ymax - k * 0.075 * (ymax - ymin),
                                        f"{electrode} = {initial_voltages[electrode]:.2f} V", ha='right', va='top'))

        ax.set_aspect('equal')
        ax.set_xlabel("$x$"+f" ({chr(956)}m)")
        ax.set_ylabel("$y$"+f" ({chr(956)}m)")
        plt.locator_params(axis='both', nbins=4)

        fig.tight_layout()

        def update(frame):
            # Update the voltages and electron positions
            voltages = list_of_voltages[frame]
            final_x, final_y = r2xy(list_of_electron_positions[frame])

            potential = make_potential(self.potential_dict, voltages)
            zdata = -potential.T

            # Update the color plot
            img_data.set_data(zdata[::-1, :])

            # Update the electron positions (green dots)
            pts_data[0].set_xdata(final_x * 1e6)
            pts_data[0].set_ydata(final_y * 1e6)

            if print_voltages:
                # Update the voltages to the left of the image
                for k, electrode in enumerate(voltages.keys()):
                    text_boxes[k].set_text(
                        f"{electrode} = {voltages[electrode]:.2f} V")

            return (img_data, pts_data, text_boxes)

        return animation.FuncAnimation(fig=fig, func=update, frames=np.arange(len(list_of_voltages)), interval=frame_interval_ms, repeat=True)

    def animate_convergence(self, fig, ax, coor: tuple = (0, 0), dxdy: tuple = (2, 2), frame_interval_ms: int = 10) -> matplotlib.animation.FuncAnimation:
        """Animate the convergence data stored in the convergence helper class. 

        Args:
            coor (tuple, optional): [microns] The coordinates of the center of the plot. Defaults to (0, 0).
            dxdy (tuple, optional): [microns] The width and height of the plot. Defaults to (2, 2).
            frame_interval_ms (int, optional): Interval between frames in milliseconds. Defaults to 10.

        Returns:
            _type_: matplotlib.animation.FuncAnimation object.
        """
        # The position data is stored in the coordinates of the helper class
        r = self.CM.curr_xk

        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        self.plot_potential_energy(
            ax=ax, coor=coor, dxdy=dxdy, print_voltages=False, plot_contours=False)

        rx, ry = r2xy(r[0, :])
        pts_data = ax.plot(rx*1e6, ry*1e6, 'ok', mfc='mediumseagreen', mew=0.5,
                           ms=10, path_effects=[pe.SimplePatchShadow(), pe.Normal()])

        # Only things in the update function will get updated.
        def update(frame):
            rx, ry = r2xy(r[frame, :])
            # Update the electron positions (green dots)
            pts_data[0].set_xdata(rx * 1e6)
            pts_data[0].set_ydata(ry * 1e6)

            return pts_data,

        fig.tight_layout()
        # The interval is in milliseconds
        return animation.FuncAnimation(fig=fig, func=update, frames=np.arange(self.CM.curr_xk.shape[0]), interval=frame_interval_ms, repeat=True)

    def plot_convergence(self, ax=None) -> None:
        """Plot the convergence of the latest solution from find_ground_configuration

        Args:
            ax (optional): Matplotlib axes object. Defaults to None.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5., 3.5))
        ax.plot(self.CM.curr_grad_norm)
        ax.set_yscale('log')
        ax.set_xlim(-1, len(self.CM.curr_grad_norm) + 1)
        ax.locator_params(axis='x', nbins=4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost function")
