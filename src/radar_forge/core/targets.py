"""Radar targets and their cross-sections.

Scenario 001 needs exactly one thing from this module: a point target with a
constant radar cross-section, following a trajectory. That is deliberately the
whole of it.

Scope
-----
Only the **Swerling 0** (equivalently Swerling 5) non-fluctuating model is
implemented. The fluctuating models Swerling 1-4 named in ``spec/structure.md``,
and the extended multi-scatterer targets alongside them, are **not implemented
here**. They are not stubbed either: an unimplemented model is better than a
wrong one, and a teaching library that ships a function whose body is ``pass``
teaches that it is acceptable to.

A non-fluctuating cross-section is the right default for a first scenario. It
makes the received power a deterministic function of geometry, so a detection
that appears in the wrong range-Doppler bin is a bug in the geometry and not an
unlucky draw.

References
----------
.. [1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed.,
       McGraw-Hill, 2014, §2.3 (radar cross-section), §7.2 (Swerling models).
.. [2] M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill,
       2001, §2.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["PointTarget", "dbsm_to_m2", "m2_to_dbsm"]


def dbsm_to_m2(rcs_dbsm: ArrayLike) -> NDArray[np.float64]:
    """Convert radar cross-section from dBsm to square metres.

    Parameters
    ----------
    rcs_dbsm : array_like
        Radar cross-section in decibels relative to one square metre.

    Returns
    -------
    numpy.ndarray
        Radar cross-section in square metres, always positive.

    See Also
    --------
    m2_to_dbsm : The inverse.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(dbsm_to_m2(10.0), 10.0))
    True
    >>> bool(np.isclose(dbsm_to_m2(0.0), 1.0))
    True
    """
    return np.asarray(10.0 ** (np.asarray(rcs_dbsm, dtype=np.float64) / 10.0), dtype=np.float64)


def m2_to_dbsm(rcs_m2: ArrayLike) -> NDArray[np.float64]:
    """Convert radar cross-section from square metres to dBsm.

    Parameters
    ----------
    rcs_m2 : array_like
        Radar cross-section in square metres. Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        Radar cross-section in dBsm.

    Raises
    ------
    ValueError
        If any cross-section is non-positive, which has no decibel value.

    See Also
    --------
    dbsm_to_m2 : The inverse.
    """
    rcs_arr = np.asarray(rcs_m2, dtype=np.float64)
    if np.any(rcs_arr <= 0.0):
        msg = "rcs_m2 must be strictly positive; a non-positive area has no decibel value."
        raise ValueError(msg)
    return np.asarray(10.0 * np.log10(rcs_arr), dtype=np.float64)


@dataclass(frozen=True)
class PointTarget:
    """A non-fluctuating point target (Swerling 0).

    The target is a single scatterer: smaller than a range bin, with one
    cross-section that does not vary with aspect angle, time, or pulse. For the
    scenario-001 aircraft at 8-18 km with 75 m range bins, the point assumption
    is sound; the aspect-independence is an approximation, and a real aircraft's
    cross-section varies by tens of decibels as it turns.

    Parameters
    ----------
    rcs_m2 : float
        Radar cross-section, square metres. 10.0 (10 dBsm) is a reasonable
        light-aircraft value.
    name : str
        A label carried through to the truth output. Default ``"target"``.

    Raises
    ------
    ValueError
        If ``rcs_m2`` is negative. Zero is allowed, and is the way to run a
        scenario with noise only.

    Notes
    -----
    Swerling 1-4 fluctuation is not implemented; see the module docstring. When
    it is, it will take an explicit ``rng`` argument rather than touching global
    NumPy random state, per ``docs/conventions/style.md`` §7.

    Examples
    --------
    >>> target = PointTarget.from_dbsm(10.0, name="light-aircraft")
    >>> round(target.rcs_m2, 6)
    10.0
    """

    rcs_m2: float
    name: str = "target"

    def __post_init__(self) -> None:
        """Validate the cross-section; see the class ``Raises`` section."""
        if self.rcs_m2 < 0.0:
            msg = (
                f"rcs_m2 must be non-negative; got {self.rcs_m2!r}. "
                "Use 0.0 for a noise-only scenario."
            )
            raise ValueError(msg)

    @classmethod
    def from_dbsm(cls, rcs_dbsm: float, name: str = "target") -> PointTarget:
        """Construct from a cross-section in dBsm, the usual way one is quoted.

        Parameters
        ----------
        rcs_dbsm : float
            Radar cross-section in dBsm.
        name : str
            A label carried through to the truth output.

        Returns
        -------
        PointTarget
            A target with the equivalent linear cross-section.
        """
        return cls(rcs_m2=float(dbsm_to_m2(rcs_dbsm)), name=name)

    @property
    def rcs_dbsm(self) -> float:
        """Cross-section in dBsm, or ``-inf`` for a zero-cross-section target."""
        if self.rcs_m2 == 0.0:
            return float("-inf")
        return float(m2_to_dbsm(self.rcs_m2))
