"""Step 1: the local slope of every pixel (Horn 1981), the input of mean_slp and slp_std.

Horn's 3 x 3 kernel on the elevation:

    dz/dx = [(z2 + 2 z5 + z8) - (z0 + 2 z3 + z6)] / (8 dx)
    dz/dy = [(z0 + 2 z1 + z2) - (z6 + 2 z7 + z8)] / (8 dy)
    slope = atan(sqrt((dz/dx / mx)^2 + (dz/dy / my)^2)) in degrees

with z0 .. z8 the window read row by row from the north-west, dx and dy the pixel size in degrees and mx, my the
metres of one degree east and north at the pixel's latitude (the fixed series of :mod:`flowmorph.earth`).  A
neighbour without an elevation takes the centre value.

The neighbours of a pixel on the edge of the grid lie outside it.  Give ``ring``, the elevation of the grid with one
more row and column on every side (from the global elevation the grid is cut from), and they are read from there; a
basin cut out of a larger grid then has the slope it has in the larger grid.  Without it they take the centre
value.
"""

import math

import numpy as np
from numba import njit

from .earth import degree_metres_x, degree_metres_y

NODATA = -9999.0


@njit
def _is_nodata(value):
    return not math.isfinite(value) or abs(value - NODATA) < 1e-6


@njit
def _slope_of_grid(elevation, ring, north_deg, x_resolution_deg, y_resolution_deg):
    nrow, ncol = elevation.shape
    slope = np.full((nrow, ncol), np.float32(NODATA), dtype=np.float32)
    mismatch_count = 0
    one_eighth = 1.0 / 8.0
    window = np.empty((3, 3), dtype=np.float32)
    for row in range(nrow):
        latitude = north_deg + (row + 0.5) * y_resolution_deg
        metres_east = degree_metres_x(latitude)
        metres_north = degree_metres_y(latitude)
        for col in range(ncol):
            centre = elevation[row, col]
            if _is_nodata(np.float64(centre)):
                continue
            if ring[row + 1, col + 1] != centre:
                mismatch_count += 1
            for window_row in range(3):
                for window_col in range(3):
                    neighbour = ring[row + window_row, col + window_col]
                    if _is_nodata(np.float64(neighbour)):
                        window[window_row, window_col] = centre
                    else:
                        window[window_row, window_col] = neighbour
            change_east = ((np.float64(window[0, 2]) + 2.0 * np.float64(window[1, 2]) + np.float64(window[2, 2])
                            - np.float64(window[0, 0]) - 2.0 * np.float64(window[1, 0]) - np.float64(window[2, 0]))
                           * (one_eighth / abs(x_resolution_deg)))
            change_north = ((np.float64(window[0, 0]) + 2.0 * np.float64(window[0, 1]) + np.float64(window[0, 2])
                             - np.float64(window[2, 0]) - 2.0 * np.float64(window[2, 1]) - np.float64(window[2, 2]))
                            * (one_eighth / abs(y_resolution_deg)))
            if not (metres_east > 0.0) or not (metres_north > 0.0):
                continue
            gradient = math.hypot(change_east / metres_east, change_north / metres_north)
            slope_deg = math.atan(gradient) * (180.0 / math.pi)
            if math.isfinite(slope_deg):
                slope[row, col] = np.float32(slope_deg)
    return slope, mismatch_count


def local_slope(elevation, transform, ring=None):
    """Horn's slope in degrees, float32, -9999 where the elevation is missing.

    Parameters
    ----------
    elevation : 2-D float array, -9999 (or not finite) where there is none
    transform : the six numbers of the grid (GDAL order), longitude and latitude, north up
    ring : optional 2-D float array of shape (nrow + 2, ncol + 2): the same grid with one more row and column on
        every side; it must equal ``elevation`` wherever ``elevation`` has a value
    """
    elevation = np.ascontiguousarray(elevation, dtype=np.float32)
    if elevation.ndim != 2:
        raise ValueError("the elevation must be a 2-D grid")
    nrow, ncol = elevation.shape
    if ring is None:
        ring = np.full((nrow + 2, ncol + 2), np.float32(NODATA), dtype=np.float32)
        ring[1:-1, 1:-1] = elevation
    else:
        ring = np.ascontiguousarray(ring, dtype=np.float32)
        if ring.shape != (nrow + 2, ncol + 2):
            raise ValueError(f"the ring must be {nrow + 2} x {ncol + 2}: the grid and one cell on every side")
    slope, mismatch_count = _slope_of_grid(elevation, ring, float(transform[3]), float(transform[1]),
                                           float(transform[5]))
    if mismatch_count:
        raise ValueError(f"{mismatch_count} pixels of the elevation differ from the ring at the same place; the ring "
                         f"is not read from where the grid lies")
    return slope


__all__ = ["local_slope", "NODATA"]
