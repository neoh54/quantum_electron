import numpy as np
from dataclasses import dataclass
from typing import Dict, Union
from numpy.typing import ArrayLike

ff_types = ['2Dmap', '2Dslices']


@dataclass(slots=True)
class CouplingConstants:
    """Coupling constants of the electrodes, compatible with zeroheliumkit.fem.CouplingConstants.
    Any object with the attributes x, y and data (such as the zeroheliumkit class) is accepted by FullModel.

    Attributes:
        x (np.ndarray): 1D array of x-coordinates in microns.
        y (np.ndarray): 1D array of y-coordinates in microns.
        data (dict): Electrode names as keys, and 2D arrays of shape (len(y), len(x)) as values, i.e. indexed as data[electrode][y, x].
    """
    x: np.ndarray
    y: np.ndarray
    data: dict


def flatten(l: list) -> list:
    return [item for sublist in l for item in sublist]


def read_ff_output(filename: str, ff_type: str) -> dict:
    """ Reads the output file and extract the data based on the specified ff_type.
        Returns a dictionary with coupling constant matricies and 'x' and 'y' lists.

    Args:
    ____
    filename (str): The path to the input file.
    ff_type (str): The type of the extracted data.
                   Choose from '2Dmap' (single) or '2Dslices' (multiple).

    Raises:
    ----
    Exception: If the specified ff_type is incorrect.
    """

    if ff_type not in ff_types:
        raise TypeError(f"Incorrect {ff_type}, choose from {ff_types}")

    if ff_type == '2Dmap':
        # handles a single 2Dmap which has multiple coupling constants
        data = {}
        with open(filename) as file:
            for line in file:
                if line[:9] == 'startDATA':
                    t = line.split()
                    key, array, dtype = (t[1], [], 'data')
                elif line[:7] == 'startXY':
                    t = line.split()
                    key, array, dtype = (t[1], [], 'xy')
                elif line[:3] == 'END':
                    if dtype == 'data':
                        data.update({key: np.asarray(array, dtype=float)})
                    elif dtype == 'xy':
                        data.update({key: np.asarray(flatten(array), dtype=float)})
                    else: pass
                elif line.split() == []:
                    pass
                else:
                    array.append([float(item) for item in line.split()])

    elif ff_type == '2Dslices':
        # handles a multiple 2Dmaps which contains multiple coupling constants
        data = {}
        with open(filename) as file:
            for line in file:
                if line[:9] == 'startDATA':
                    t = line.split()
                    key, array, dtype = (t[1], {}, 'data')
                elif line[:12] == 'start2DSLICE':
                    t = line.split()
                    s_key, s_array = (t[1], [])
                elif line[:7] == 'startXY':
                    t = line.split()
                    key, s_array, dtype = (t[1], [], 'xy')
                elif line[:3] == 'end':
                    array.update({s_key: np.asarray(s_array, dtype=float)})
                elif line[:3] == 'END':
                    if dtype == 'data':
                        data.update({key: array})
                    elif dtype == 'xy':
                        data.update({key: np.asarray(flatten(s_array), dtype=float)})
                    else: pass
                elif line.split() == []:
                    pass
                else:
                    s_array.append([float(item) for item in line.split()])
    else: pass
    return data


def load_coupling_constants(filename: str) -> CouplingConstants:
    """Loads a FreeFem '2Dmap' output file (as in examples/fem_data) into a CouplingConstants object.

    Args:
        filename (str): Path to the FreeFem output file.

    Returns:
        CouplingConstants: Coupling constants with arrays indexed as data[electrode][y, x].
    """
    raw = read_ff_output(filename, '2Dmap')
    # The FreeFem file stores the arrays indexed as [x, y], CouplingConstants uses [y, x].
    return CouplingConstants(x=raw['xlist'], y=raw['ylist'],
                             data={k: v.T for k, v in raw.items() if k not in ['xlist', 'ylist']})


def to_potential_dict(potential: Union[Dict[str, ArrayLike], CouplingConstants]) -> Dict[str, ArrayLike]:
    """Converts a CouplingConstants object (from this package or zeroheliumkit) into the potential_dict used internally:
    electrode names as keys with 2D arrays indexed as [x, y], plus the keys 'xlist' and 'ylist' (microns).
    A potential_dict is returned unchanged.

    Args:
        potential (Union[Dict[str, ArrayLike], CouplingConstants]): CouplingConstants object or potential_dict.

    Returns:
        Dict[str, ArrayLike]: potential_dict
    """
    if isinstance(potential, dict):
        return potential

    if not all(hasattr(potential, attr) for attr in ['x', 'y', 'data']):
        raise TypeError("Expected a potential_dict or a CouplingConstants object with attributes x, y and data.")

    x, y = np.asarray(potential.x, dtype=float), np.asarray(potential.y, dtype=float)
    potential_dict = {}
    for electrode, array in potential.data.items():
        if np.ma.is_masked(array):
            raise ValueError(f"The coupling constants of '{electrode}' contain masked values, which cannot be interpolated. "
                             "Crop the coupling constants instead of masking them.")
        array = np.asarray(np.ma.getdata(array), dtype=float)
        if array.shape != (len(y), len(x)):
            raise ValueError(f"The coupling constants of '{electrode}' have shape {array.shape}, "
                             f"expected (len(y), len(x)) = {(len(y), len(x))}.")
        potential_dict[electrode] = array.T

    potential_dict['xlist'] = x
    potential_dict['ylist'] = y
    return potential_dict
