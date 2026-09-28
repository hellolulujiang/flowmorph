"""GeoTIFF in and out, through rasterio.

A grid comes back as its array, the six numbers of its transform (GDAL order) and its CRS.  Cells a file marks
invalid (its nodata value, or an internal mask) are set to the value FlowMorph reads as no data: 247 for the flow
directions, -9999 for a float grid.
"""

import numpy as np
import rasterio
from rasterio.windows import Window

FLOAT_NODATA = -9999.0
D8_NODATA = 247


def _check_geographic(dataset, path):
    crs = dataset.crs
    if crs is None or not crs.is_geographic:
        raise ValueError(f"{path}: FlowMorph reads longitude-latitude grids in degrees (the lengths and areas "
                         f"are taken on the earth from the pixel's latitude); this grid has CRS {crs}")
    transform = dataset.transform
    if transform.b != 0.0 or transform.d != 0.0 or not transform.a > 0.0 or not transform.e < 0.0:
        raise ValueError(f"{path}: the grid must be north up with no rotation")


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
        d8[mask == 0] = D8_NODATA
        return d8, tuple(dataset.transform.to_gdal()), dataset.crs


def read_float(path, shape=None, transform=None):
    """A float grid as float32 with -9999 where the file has none; checked against a shape and a transform."""
    with rasterio.open(path) as dataset:
        _check_geographic(dataset, path)
        values = dataset.read(1).astype(np.float32)
        mask = dataset.read_masks(1)
        values[(mask == 0) | ~np.isfinite(values)] = np.float32(FLOAT_NODATA)
        if shape is not None and values.shape != tuple(shape):
            raise ValueError(f"{path} is {values.shape[0]} x {values.shape[1]}, the flow directions "
                             f"{shape[0]} x {shape[1]}")
        if transform is not None and not _same_grid(dataset.transform.to_gdal(), transform):
            raise ValueError(f"{path} does not lie on the grid of the flow directions")
        return values


def read_int(path, shape=None, transform=None):
    """An integer grid (seq_dfs, upg) as int64, with -1 where the file has none."""
    with rasterio.open(path) as dataset:
        if np.dtype(dataset.dtypes[0]).kind not in "iu":
            raise ValueError(f"{path}: an index must be an integer raster")
        values = dataset.read(1).astype(np.int64)
        mask = dataset.read_masks(1)
        values[mask == 0] = -1
        if shape is not None and values.shape != tuple(shape):
            raise ValueError(f"{path} does not have the shape of the flow directions")
        if transform is not None and not _same_grid(dataset.transform.to_gdal(), transform):
            raise ValueError(f"{path} does not lie on the grid of the flow directions")
        return values


def _same_grid(first, second):
    size = abs(second[1])
    return (abs(first[1] - second[1]) <= 1e-9 * size and abs(first[5] - second[5]) <= 1e-9 * size
            and abs(first[0] - second[0]) <= 1e-3 * size and abs(first[3] - second[3]) <= 1e-3 * size)


def read_ring(path, shape, transform):
    """The elevation of the grid with one more row and column on every side, from a larger elevation grid (the
    global one the grid was cut from), as float32 with -9999 where there is none.  Longitude wraps round when the
    larger grid spans 360 degrees; a row beyond its top or bottom stays no data."""
    nrow, ncol = shape
    with rasterio.open(path) as dataset:
        _check_geographic(dataset, path)
        big = dataset.transform.to_gdal()
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
            block = dataset.read(1, window=window).astype(np.float32)
            block_mask = dataset.read_masks(1, window=window)
            block[(block_mask == 0) | ~np.isfinite(block)] = np.float32(FLOAT_NODATA)
            ring[ring_rows, run[0]:run[-1] + 1] = block
        return ring


def write_float(path, values, transform, crs):
    """A float32 GeoTIFF, -9999 no data, tiled and compressed."""
    values = np.asarray(values, dtype=np.float32)
    profile = {
        "driver": "GTiff", "height": values.shape[0], "width": values.shape[1], "count": 1, "dtype": "float32",
        "crs": crs, "transform": rasterio.Affine.from_gdal(*transform), "nodata": FLOAT_NODATA,
        "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate", "predictor": 3,
    }
    if values.shape[0] < 256 or values.shape[1] < 256:
        profile.update(tiled=False)
        profile.pop("blockxsize")
        profile.pop("blockysize")
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values, 1)


__all__ = ["read_d8", "read_float", "read_int", "read_ring", "write_float", "FLOAT_NODATA", "D8_NODATA"]
