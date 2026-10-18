"""Reconstruction surfaces: occupancy grid and distance field (TARGET_ARCHITECTURE §6.1, track C4).

Everything here lives in the Site ENU frame (metres, x east / y north / z up) and is plain
numpy/scipy — no torch, no GPU — like `aerosplat.semantics`, which it builds on. The grid
and its queries are in `aerosplat.surface.occupancy`; this package root stays light, because
the pipeline registry imports it.
"""

from aerosplat.surface.models import (FREE, OCCUPIED, OUTSIDE, UNKNOWN, ColumnSource, GridSpec,
                                      OccupancyReport)

__all__ = ["FREE", "OCCUPIED", "OUTSIDE", "UNKNOWN", "ColumnSource", "GridSpec", "OccupancyReport"]
