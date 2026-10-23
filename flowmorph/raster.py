"""GeoTIFF in and out, through rasterio.

A grid comes back as its array, the six numbers of its transform (GDAL order) and its CRS.  Pixels a file marks
invalid (its declared no-data value) are set to the value FlowMorph reads as no data: 247 for the flow directions,
-9999 for a float grid, -1 for an index.  A mask band or an alpha band may hide only such pixels: one that hides
pixels holding values is refused, as the C programs would read those values.
"""

import math
import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window

from .grid import snap_transform

FLOAT_NODATA = -9999.0
D8_NODATA = 247


def _check_crs(crs, path, expected_crs=None):
    crs = None if crs is None else CRS.from_user_input(crs)
    if crs is None or not crs.is_geographic:
        raise ValueError(f"{path}: FlowMorph reads longitude-latitude grids in degrees (the lengths and areas "
                         f"are taken on the earth from the pixel's latitude); this grid has CRS {crs}")
    unit_name, radians_per_unit = crs.units_factor
    if not math.isclose(radians_per_unit, math.pi / 180.0, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"{path}: FlowMorph requires angular units in degrees, not {unit_name}")
    if expected_crs is not None and crs != CRS.from_user_input(expected_crs):
        raise ValueError(f"{path} does not have the same CRS as the flow directions")


def _check_geographic(dataset, path, expected_crs=None):
    _check_crs(dataset.crs, path, expected_crs)
    transform = dataset.transform
    if not all(math.isfinite(value) for value in transform.to_gdal()):
        raise ValueError(f"{path}: the transform must be six finite numbers")
    if transform.b != 0.0 or transform.d != 0.0 or not transform.a > 0.0 or not transform.e < 0.0:
        raise ValueError(f"{path}: the grid must be north up with no rotation")


def _refuse_a_mask_over_values(path, dataset, values, mask, own_nodata):
    """A file's mask may hide only pixels that hold no data: the file's declared no-data value, or ``own_nodata``,
    the value this package and the C code take for no data in that grid (247 in the directions, -9999 in a float
    grid, -1 in an index).  The C programs read the values and know no mask: an internal mask band or an alpha band
    over pixels that hold values would give this package another network, or other attributes, than the C code reads
    from the same file, without a word.  A file whose mask is its declared no-data value, as
    MERIT Hydro's are, is read as its values say."""
    hidden = mask == 0
    if not hidden.any():
        return
    holds_a_value = np.ones(values.shape, dtype=bool)
    if values.dtype.kind == "f":
        holds_a_value &= np.isfinite(values)
    for nodata in (dataset.nodata, own_nodata):
        if nodata is None or nodata != nodata:                  # none declared, or NaN (taken out above)
            continue
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                declared = np.array(nodata).astype(values.dtype)
            # the value as the raster's type holds it: a float no-data such as 0.1 is not the same double once it
            # is a float32, and is the raster's no-data all the same; a whole-number type must hold it exactly
            if (values.dtype.kind == "f" and np.isfinite(declared)) or float(declared) == float(nodata):
                holds_a_value &= values != declared
        except (OverflowError, ValueError):
            pass
    if np.any(hidden & holds_a_value):
        raise ValueError(f"{path}: its mask hides {int(np.count_nonzero(hidden & holds_a_value))} pixels that hold "
                         "values. The values are what the C programs read, so the two would not agree: write the "
                         "no-data value into those pixels, or remove the mask")


def _holds_the_declared_no_data(dataset, values):
    """The pixels that hold the file's declared no-data value, whatever a mask says of them.  This package judges by
    the value, as the C programs do with theirs (-9999, 247): a mask band may call valid a pixel that holds the
    declared value, and it would be read as an elevation, an area or a rank.  A file whose mask is meant to give
    such a pixel back as data is therefore not read as its mask says: the value must not be used for data.  None
    when the file declares no value, declares NaN (not finite: taken out as such) or declares a value its own type
    cannot hold."""
    nodata = dataset.nodata
    if nodata is None or nodata != nodata:
        return None
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            declared = np.array(nodata).astype(values.dtype)
    except (OverflowError, ValueError):
        return None
    if not ((values.dtype.kind == "f" and np.isfinite(declared)) or float(declared) == float(nodata)):
        return None
    return values == declared


def read_d8(path):
    """The MERIT D8 grid (uint8, 247 no data), its transform and its CRS."""
    with rasterio.open(path) as dataset:
        _check_geographic(dataset, path)
        if dataset.dtypes[0] != "uint8":
            raise ValueError(f"{path}: the flow directions must be a Byte raster, not {dataset.dtypes[0]}")
        if dataset.nodata is not None and dataset.nodata != D8_NODATA:
            raise ValueError(f"{path}: the flow directions declare the no data {dataset.nodata}; MERIT's is 247")
        d8 = dataset.read(1)
        mask = dataset.read_masks(1)
        _refuse_a_mask_over_values(path, dataset, d8, mask, D8_NODATA)
        d8[mask == 0] = D8_NODATA
        return d8, tuple(dataset.transform.to_gdal()), dataset.crs


def read_float(path, shape=None, transform=None, expected_crs=None):
    """A float grid as float32 with -9999 where absent; check the supplied shape, transform and optional CRS."""
    with rasterio.open(path) as dataset:
        _check_geographic(dataset, path, expected_crs)
        values = dataset.read(1)
        mask = dataset.read_masks(1)
        _refuse_a_mask_over_values(path, dataset, values, mask, FLOAT_NODATA)
        absent = mask == 0
        declared = _holds_the_declared_no_data(dataset, values)
        if declared is not None:
            absent |= declared
        values = values.astype(np.float32)
        values[absent | ~np.isfinite(values)] = np.float32(FLOAT_NODATA)
        if shape is not None and values.shape != tuple(shape):
            raise ValueError(f"{path} is {values.shape[0]} x {values.shape[1]}, the flow directions "
                             f"{shape[0]} x {shape[1]}")
        if transform is not None and not _same_grid(dataset.transform.to_gdal(), transform):
            raise ValueError(f"{path} does not lie on the grid of the flow directions")
        return values


def read_int(path, shape=None, transform=None, expected_crs=None):
    """A geographic integer grid (seq_dfs, upg) as int64, -1 where absent; check the optional reference grid/CRS."""
    with rasterio.open(path) as dataset:
        _check_geographic(dataset, path, expected_crs)
        if np.dtype(dataset.dtypes[0]).kind not in "iu":
            raise ValueError(f"{path}: an index must be an integer raster")
        values = dataset.read(1)
        mask = dataset.read_masks(1)
        _refuse_a_mask_over_values(path, dataset, values, mask, -1)
        absent = mask == 0
        declared = _holds_the_declared_no_data(dataset, values)
        if declared is not None:
            absent |= declared
        values = values.astype(np.int64)
        values[absent] = -1
        if shape is not None and values.shape != tuple(shape):
            raise ValueError(f"{path} does not have the shape of the flow directions")
        if transform is not None and not _same_grid(dataset.transform.to_gdal(), transform):
            raise ValueError(f"{path} does not lie on the grid of the flow directions")
        return values


def _same_grid(first, second):
    if len(first) != 6 or len(second) != 6 or not all(math.isfinite(value) for value in (*first, *second)):
        return False
    # as the C code takes them: two headers that write one grid with different rounding (1/1200 and 0.00083333) are
    # one grid
    first = snap_transform(first)
    second = snap_transform(second)
    size = abs(second[1])
    return all(abs(first[index] - second[index]) <= (1e-3 if index in (0, 3) else 1e-9) * size
               for index in range(6))


def read_ring(path, shape, transform, expected_crs=None):
    """The elevation of the grid with one more row and column on every side, from a larger elevation grid (the
    global one the grid was cut from), as float32 with -9999 where there is none.  Longitude wraps round when the
    larger grid spans 360 degrees; a row beyond its top or bottom stays no data. When supplied, ``expected_crs``
    must match the larger grid's CRS."""
    nrow, ncol = shape
    with rasterio.open(path) as dataset:
        _check_geographic(dataset, path, expected_crs)
        big = snap_transform(dataset.transform.to_gdal())
        transform = snap_transform(transform)
        if abs(big[1] - transform[1]) > 1e-9 * abs(transform[1]) or \
                abs(big[5] - transform[5]) > 1e-9 * abs(transform[1]):
            raise ValueError(f"{path} does not have the pixel size of the grid")
        col_offset = (transform[0] - big[0]) / big[1]
        row_offset = (transform[3] - big[3]) / big[5]
        first_col = int(round(col_offset))
        first_row = int(round(row_offset))
        if abs(col_offset - first_col) > 1e-3 or abs(row_offset - first_row) > 1e-3:
            raise ValueError(f"the grid does not lie on the pixels of {path}")
        spans_globe = abs(dataset.width * big[1] - 360.0) < 1e-6
        ring = np.full((nrow + 2, ncol + 2), np.float32(FLOAT_NODATA), dtype=np.float32)
        top = max(first_row - 1, 0)
        bottom = min(first_row + nrow, dataset.height - 1)
        if bottom < top:
            return ring
        cols = np.arange(first_col - 1, first_col + ncol + 1)
        if spans_globe:
            cols = np.mod(cols, dataset.width)
        inside = np.flatnonzero((cols >= 0) & (cols < dataset.width))
        ring_rows = slice(top - (first_row - 1), bottom - (first_row - 1) + 1)
        # one read per run of consecutive columns: a ring across the antimeridian is two runs, one at each end of
        # the larger grid, and reading from the lowest column to the highest would read its whole width
        run_starts = np.flatnonzero(np.diff(cols[inside]) != 1) + 1
        for run in np.split(inside, run_starts):
            if run.size == 0:
                continue
            first = int(cols[run[0]])
            window = Window(first, top, run.size, bottom - top + 1)
            block = dataset.read(1, window=window)
            block_mask = dataset.read_masks(1, window=window)
            _refuse_a_mask_over_values(path, dataset, block, block_mask, FLOAT_NODATA)
            absent = block_mask == 0
            declared = _holds_the_declared_no_data(dataset, block)
            if declared is not None:
                absent |= declared
            block = block.astype(np.float32)
            block[absent | ~np.isfinite(block)] = np.float32(FLOAT_NODATA)
            ring[ring_rows, run[0]:run[-1] + 1] = block
        return ring


def write_float(path, values, transform, crs):
    """A float32 GeoTIFF, -9999 no data, compressed; in tiles of 256 when both sides are 256 pixels or more, in
    strips otherwise."""
    values = np.asarray(values, dtype=np.float32)
    profile = {
        "driver": "GTiff", "height": values.shape[0], "width": values.shape[1], "count": 1, "dtype": "float32",
        "crs": crs, "transform": rasterio.Affine.from_gdal(*transform), "nodata": FLOAT_NODATA,
        "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate", "predictor": 3,
        "BIGTIFF": "IF_SAFER",              # a large grid that compresses badly passes the 4 GiB of a classic TIFF
    }
    if values.shape[0] < 256 or values.shape[1] < 256:
        profile.update(tiled=False)
        profile.pop("blockxsize")
        profile.pop("blockysize")
    if os.path.lexists(path):
        raise FileExistsError(f"{path} exists and is not written over: move it away or give another directory")
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values, 1)


__all__ = ["read_d8", "read_float", "read_int", "read_ring", "write_float", "FLOAT_NODATA", "D8_NODATA"]
