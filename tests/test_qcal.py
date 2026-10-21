"""Harmonic inversion used by the loss-port Q calibration."""
import numpy as np

from picstudio.core.qcal import harmonic_inversion


def test_two_close_modes():
    # pi mode (Q = 250) next to a high-Q mode 70 MHz below, as in a 6-cavity planar magnetron
    dt = 1.36e-13 * 20
    t = np.arange(5500) * dt
    x = (np.exp(-np.pi * 9.053e9 / 250 * t) * np.cos(2 * np.pi * 9.053e9 * t + 0.3)
         + 0.7 * np.exp(-np.pi * 8.983e9 / 1500 * t) * np.cos(2 * np.pi * 8.983e9 * t + 1.1))
    modes = harmonic_inversion(x, dt, 8.6e9, 9.5e9, max_modes=4)
    pi = min(modes, key=lambda m: abs(m["f"] - 9.053e9))
    assert abs(pi["f"] / 9.053e9 - 1) < 1e-4
    assert abs(pi["Q"] / 250 - 1) < 0.02
    other = min(modes, key=lambda m: abs(m["f"] - 8.983e9))
    assert abs(other["Q"] / 1500 - 1) < 0.05
