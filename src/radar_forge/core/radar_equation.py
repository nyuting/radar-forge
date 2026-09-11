"""The monostatic radar range equation.

This module is the worked example for the conventions in
``docs/conventions/style.md``: SI units with unit-suffixed names, NumPy-style
docstrings that cite their source, complete type annotations, and vectorised
NumPy rather than Python loops.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §2.2 (eq. 2.11).
.. [2] L. Harrison and G. Andrews, *Introduction to Radar Using Python and
       MATLAB*, Artech House, 2020, ch. 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["received_power_w"]

_FOUR_PI_CUBED = (4.0 * np.pi) ** 3


def received_power_w(
    transmit_power_w: ArrayLike,
    gain_tx_linear: ArrayLike,
    gain_rx_linear: ArrayLike,
    wavelength_m: ArrayLike,
    rcs_m2: ArrayLike,
    range_m: ArrayLike,
    loss_linear: ArrayLike = 1.0,
) -> NDArray[np.float64]:
    r"""Return the power received from a point target, in watts.

    Implements the monostatic radar range equation [1]_

    .. math::

        P_r = \frac{P_t\, G_t\, G_r\, \lambda^2\, \sigma}{(4\pi)^3\, R^4\, L}

    All arguments are linear (not decibel) SI quantities and are broadcast
    against one another, so any argument may be a scalar or an array.

    Parameters
    ----------
    transmit_power_w : array_like
        Peak transmitted power :math:`P_t`, in watts.
    gain_tx_linear, gain_rx_linear : array_like
        Transmit and receive antenna gains :math:`G_t`, :math:`G_r`, linear
        (a 30 dBi antenna is ``10 ** (30 / 10)``, not ``30``).
    wavelength_m : array_like
        Carrier wavelength :math:`\lambda`, in metres.
    rcs_m2 : array_like
        Target radar cross-section :math:`\sigma`, in square metres. Convert
        from dBsm with ``10 ** (rcs_dbsm / 10)``.
    range_m : array_like
        Range :math:`R` from the radar to the target, in metres. Must be
        strictly positive.
    loss_linear : array_like, optional
        Aggregate loss factor :math:`L \ge 1` (atmospheric, system, processing),
        linear rather than dB. Default 1.0, i.e. lossless.

    Returns
    -------
    numpy.ndarray
        Received power in watts, of the broadcast shape of the inputs.

    Raises
    ------
    ValueError
        If any range is non-positive, or any loss factor is less than one.

    Notes
    -----
    Received power falls as :math:`1/R^4`: doubling the range costs 12 dB.

    Examples
    --------
    >>> import numpy as np
    >>> p = received_power_w(
    ...     transmit_power_w=1.0,
    ...     gain_tx_linear=10 ** 3.0,
    ...     gain_rx_linear=10 ** 3.0,
    ...     wavelength_m=3.9e-3,
    ...     rcs_m2=10.0,
    ...     range_m=100.0,
    ... )
    >>> bool(np.isclose(p, received_power_w(1.0, 1e3, 1e3, 3.9e-3, 10.0, 200.0) * 16.0))
    True
    """
    range_arr = np.asarray(range_m, dtype=np.float64)
    loss_arr = np.asarray(loss_linear, dtype=np.float64)

    if np.any(range_arr <= 0.0):
        msg = "range_m must be strictly positive; the range equation diverges at R = 0."
        raise ValueError(msg)
    if np.any(loss_arr < 1.0):
        msg = "loss_linear must be a linear factor >= 1.0 (1.0 means lossless), not a gain."
        raise ValueError(msg)

    numerator = (
        np.asarray(transmit_power_w, dtype=np.float64)
        * np.asarray(gain_tx_linear, dtype=np.float64)
        * np.asarray(gain_rx_linear, dtype=np.float64)
        * np.asarray(wavelength_m, dtype=np.float64) ** 2
        * np.asarray(rcs_m2, dtype=np.float64)
    )
    denominator = _FOUR_PI_CUBED * range_arr**4 * loss_arr
    result: NDArray[np.float64] = numerator / denominator
    return result
