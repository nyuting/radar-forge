r"""Closed-form trajectories: the true path of a target, evaluated exactly at any time.

A target's truth is an :class:`Arc`, a start state and a chain of :class:`Leg` s: straight with
constant acceleration along track, or a turn at constant rate and speed. Headings are clockwise
from north, a turn rate positive to the right; positions are ENU metres about the radar. The
plot-level simulator (:mod:`radar_forge.plot_simulator`) and the scoring of scenario 011 read
them; MATLAB calls one a waypoint trajectory, Stone Soup a ground-truth path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Arc", "Leg"]


def _heading(heading_rad: ArrayLike) -> NDArray[np.float64]:
    """Return the unit ENU vector of a heading clockwise from north, ``(..., 2)``."""
    angle = np.asarray(heading_rad, dtype=np.float64)
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1)


@dataclass(frozen=True)
class Leg:
    """One piece of a trajectory: straight (accelerating along track, or not) or a turn."""

    duration_s: float
    turn_rate_radps: float = 0.0
    acceleration_mps2: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a leg of no duration, or one that both turns and accelerates."""
        if not (math.isfinite(self.duration_s) and self.duration_s > 0) or (
            self.turn_rate_radps != 0 and self.acceleration_mps2 != 0
        ):
            msg = f"a leg needs a duration > 0 and must not both turn and accelerate; got {self}."
            raise ValueError(msg)


@dataclass(frozen=True, eq=False)
class Arc:
    r"""A closed-form trajectory: a start state, then its legs, then straight on for ever.

    Straight: :math:`p = p_0 + (v_0 \tau + a \tau^2/2) h(\psi_0)`; turning: :math:`\psi =
    \psi_0 + \omega \tau`, :math:`p = p_0 + (v/\omega)(\cos\psi_0 - \cos\psi, \sin\psi -
    \sin\psi_0)`, :math:`h(\psi) = (\sin\psi, \cos\psi)` east and north. Before
    ``start_time_s`` the target does not exist and every value is NaN.

    >>> arc = Arc(0.0, (0.0, 0.0), 0.0, 72.0, (Leg(30.0, math.radians(3.0)),))
    >>> position_m, velocity_mps = arc.state(30.0)
    >>> np.round(position_m, 3), np.round(np.abs(velocity_mps), 6)
    (array([1375.099, 1375.099]), array([72.,  0.]))
    """

    start_time_s: float
    position_m: tuple[float, float]
    heading_rad: float
    speed_mps: float
    legs: tuple[Leg, ...] = ()
    _starts: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Find each leg's start time, position, heading and speed, ``(n_legs + 1, 4)``."""
        values = [self.start_time_s, *self.position_m, self.heading_rad, self.speed_mps]
        if len(self.position_m) != 2 or not np.all(np.isfinite(values)) or self.speed_mps < 0:
            msg = f"an arc needs a finite start, a 2-D position and a speed >= 0; got {self}."
            raise ValueError(msg)
        rows = [[self.start_time_s, *self.position_m, self.heading_rad, self.speed_mps]]
        # A chain: each leg starts where the one before ended.
        for k, leg in enumerate(self.legs):
            time_s, east, north, heading, speed = rows[-1]
            if speed + leg.acceleration_mps2 * leg.duration_s < 0:
                msg = (
                    f"leg {k} decelerates at {leg.acceleration_mps2} m/s^2 for "
                    f"{leg.duration_s} s from {speed:.3g} m/s, past a stop: the target would fly "
                    "backwards. Shorten the leg or end it at zero speed."
                )
                raise ValueError(msg)
            end = self._move(
                np.array([leg.duration_s]),
                np.array([[east, north, heading, speed]]),
                np.array([leg.turn_rate_radps]),
                np.array([leg.acceleration_mps2]),
            )
            rows.append([time_s + leg.duration_s, *end[0]])
        object.__setattr__(self, "_starts", np.array(rows))

    @staticmethod
    def _move(
        tau_s: NDArray[np.float64],
        start: NDArray[np.float64],
        turn_radps: NDArray[np.float64],
        acceleration_mps2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return east, north, heading and speed ``(n, 4)`` after ``tau_s`` on each leg."""
        east, north, heading, speed = start.T
        turning = turn_radps != 0
        rate = np.where(turning, turn_radps, 1.0)
        along = speed * tau_s + acceleration_mps2 * tau_s**2 / 2
        new_heading = heading + turn_radps * tau_s
        straight = np.stack([east, north], -1) + along[:, None] * _heading(heading)
        arc = np.stack(
            [np.cos(heading) - np.cos(new_heading), np.sin(new_heading) - np.sin(heading)], -1
        )
        turned = np.stack([east, north], -1) + (speed / rate)[:, None] * arc
        position = np.where(turning[:, None], turned, straight)
        return np.column_stack([position, new_heading, speed + acceleration_mps2 * tau_s])

    def state(self, time_s: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return position and velocity at each time, ``(..., 2)`` each, NaN before the start."""
        times = np.asarray(time_s, dtype=np.float64)
        flat = times.reshape(-1)
        leg = np.clip(np.searchsorted(self._starts[:, 0], flat, side="right") - 1, 0, None)
        turn = np.array([g.turn_rate_radps for g in self.legs] + [0.0])[leg]
        acceleration = np.array([g.acceleration_mps2 for g in self.legs] + [0.0])[leg]
        moved = self._move(flat - self._starts[leg, 0], self._starts[leg, 1:], turn, acceleration)
        moved[flat < self.start_time_s] = np.nan
        velocity = moved[:, 3:4] * _heading(moved[:, 2])
        return moved[:, :2].reshape(*times.shape, 2), velocity.reshape(*times.shape, 2)
