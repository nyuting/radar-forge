#!/usr/bin/env python3
"""Translate the flight track from its original anchor to Raleigh-Durham.

`data/flight_coordinates.csv` is a real ADS-B track of a light aircraft in the
circuit of an airport. This script re-anchors the repository's geography
onto Raleigh-Durham International Airport (RDU) and a receiver site at Duke
University in Durham, North Carolina.

The track is **translated, not synthesised**: a single constant
offset in latitude and longitude is added to every fix, so that the track's
shape, its inter-fix timing, its altitudes and the ADS-B position noise are all
preserved bit for bit. Only the anchor moves. Every tolerance in the test suite
therefore keeps the meaning it was chosen for.

The offset carries the *airport* onto RDU, because the track is airport traffic:
anchoring it anywhere else would put a circuit pattern over open country.

    offset = RDU airport reference point - original airport reference point

Running this script a second time translates again, so it reads the original
coordinates from git rather than from the working tree::

    git show <commit>:data/flight_coordinates.csv | \
        python3 scripts/translate_flight_coordinates.py > data/flight_coordinates.csv

With no redirection it reads stdin and writes stdout, so it composes with the
golden fixture the same way.

The original track has the header ``timestamp,lat,lon``. The output uses the
``spec/data-001-formats.md`` §6.3 names, ``time_s,latitude_deg,longitude_deg``,
which is what :func:`radar_forge.pipelines.trajectories.load_flight_csv` reads.
``time_s`` is seconds from the first fix, so the shipped track has no epoch. The
original track's first fix was at ``2026-09-03T00:17:56Z``.

References
----------
.. [1] FAA NASR airport record, RDU: airport reference point
       35-52-39.5000N / 078-47-14.9000W (FAA data effective 1 Oct 2026). The
       anchor used here, 35-52-39N / 078-47-15W, is within about 16 m of it.
.. [2] ``spec/data-001-formats.md`` §6.3, the trajectory columns written.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime

__all__ = [
    "COLUMNS",
    "DESTINATION_AIRPORT_LATITUDE_DEG",
    "DESTINATION_AIRPORT_LONGITUDE_DEG",
    "OFFSET_LATITUDE_DEG",
    "OFFSET_LONGITUDE_DEG",
    "SOURCE_AIRPORT_LATITUDE_DEG",
    "SOURCE_AIRPORT_LONGITUDE_DEG",
    "translate_rows",
]

#: The airport the original track was flown around.
SOURCE_AIRPORT_LATITUDE_DEG = 1.3592
SOURCE_AIRPORT_LONGITUDE_DEG = 103.9894

#: Raleigh-Durham International Airport, 35-52-39N 078-47-15W: within about 16 m
#: of the FAA airport reference point.
DESTINATION_AIRPORT_LATITUDE_DEG = 35.87750
DESTINATION_AIRPORT_LONGITUDE_DEG = -78.78750

#: The one offset, applied to every coordinate in the repository.
OFFSET_LATITUDE_DEG = DESTINATION_AIRPORT_LATITUDE_DEG - SOURCE_AIRPORT_LATITUDE_DEG
OFFSET_LONGITUDE_DEG = DESTINATION_AIRPORT_LONGITUDE_DEG - SOURCE_AIRPORT_LONGITUDE_DEG


#: The output header, per ``spec/data-001-formats.md`` §6.3.
COLUMNS = ("time_s", "latitude_deg", "longitude_deg")


def _format_degrees(degrees: float) -> str:
    """Render one coordinate to five decimal places, trailing zeros stripped."""
    return f"{degrees:.5f}".rstrip("0").rstrip(".")


def _format_seconds(seconds: float) -> str:
    """Render a time offset to three decimal places, trailing zeros stripped.

    The source timestamps are whole seconds, so in practice this writes an integer.
    """
    return f"{seconds:.3f}".rstrip("0").rstrip(".")


def translate_rows(rows: Iterable[dict[str, str]]) -> Iterator[dict[str, str]]:
    """Add the constant offset to every fix, preserving the recorded precision.

    Parameters
    ----------
    rows : iterable of dict of str to str
        Rows of the original track as :class:`csv.DictReader` yields them, with
        its ``timestamp``, ``lat`` and ``lon`` keys.

    Yields
    ------
    dict of str to str
        The same fixes under the :data:`COLUMNS` names, with latitude and
        longitude shifted. ``time_s`` is the time since the first fix, so the
        inter-fix timing is exactly the timing that was flown.

    Notes
    -----
    The source fixes carry five decimal places -- about 1.1 m, which is finer
    than ADS-B position noise -- so the output is rounded back to five places.
    Keeping full binary precision would write digits the measurement never had
    and would make the file's noise floor look better than the sensor's.
    """
    # A loop, not an array: the rows stream from stdin and stay strings throughout.
    first_fix_utc: datetime | None = None
    for row in rows:
        fix_utc = datetime.fromisoformat(row["timestamp"])
        if first_fix_utc is None:
            first_fix_utc = fix_utc
        yield {
            "time_s": _format_seconds((fix_utc - first_fix_utc).total_seconds()),
            "latitude_deg": _format_degrees(float(row["lat"]) + OFFSET_LATITUDE_DEG),
            "longitude_deg": _format_degrees(float(row["lon"]) + OFFSET_LONGITUDE_DEG),
        }


def main() -> int:
    """Translate a flight-coordinates CSV from stdin to stdout."""
    reader = csv.DictReader(sys.stdin)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(translate_rows(reader))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
