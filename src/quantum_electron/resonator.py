"""Resonator description for the dispersive coupling of electron modes to a resonator
(FullModel.get_coupling_to_mode, FullModel.get_susceptibility, FullModel.get_frequency_shift)."""
import numpy as np
from dataclasses import dataclass
from typing import Literal

resonator_modes = ['diff', 'single']


@dataclass(frozen=True)
class Resonator:
    """A single resonator mode that couples to the in-plane electron modes.

    The coupling to electron mode n is g_n = c e (E . x_n) / (2 sqrt(m_e C)), where E is the RF field per volt at the electron
    positions, x_n the normalized eigenvector, C the capacitance below and c = sqrt(2) for the differential mode, c = 1 for a
    single-ended resonator.

    Attributes:
        frequency (float): [Hz] Bare resonance frequency f_r of the resonator mode.
        capacitance (float): [F] For mode='diff': C_d, the capacitance of each node to ground plus twice the capacitance between
            the nodes (Ca + 2 Cdot in the symmetric coupled LC model of EOMSolver.setup_eom_coupled_lc). For mode='single': the
            capacitance C of the LC resonator.
        mode (str): 'diff' for the differential mode of two nodes, whose RF field is evaluated with +0.5 V / -0.5 V on two
            electrodes, or 'single' for a single-ended resonator, whose RF field is evaluated with +1 V on one electrode.
        electrodes (tuple[str, ...]): RF electrodes of the resonator: (plus, minus) for 'diff', (signal,) for 'single'.
            Defaults to (), in which case the RF field set earlier with FullModel.set_rf_interpolator is used.
    """
    frequency: float
    capacitance: float
    mode: Literal['diff', 'single']
    electrodes: tuple = ()

    def __post_init__(self):
        if self.mode not in resonator_modes:
            raise ValueError(f"mode = {self.mode!r} was not understood. Please specify one of {resonator_modes}.")
        if not (self.frequency > 0 and self.capacitance > 0):
            raise ValueError("The frequency and capacitance of the resonator must be positive.")
        # Store the electrodes as a tuple, so that the dataclass stays hashable and immutable
        object.__setattr__(self, 'electrodes', tuple(self.electrodes))
        n_electrodes = {'diff': 2, 'single': 1}[self.mode]
        if len(self.electrodes) not in [0, n_electrodes]:
            raise ValueError(f"A resonator with mode={self.mode!r} has {n_electrodes} RF electrode(s), got {self.electrodes}.")

    @property
    def mode_factor(self) -> float:
        """Factor c in the coupling g_n: sqrt(2) for the differential mode, 1 for a single-ended resonator."""
        return np.sqrt(2) if self.mode == 'diff' else 1.0
