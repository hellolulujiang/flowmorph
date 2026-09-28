"""The FlowMorph object: one grid of whole basins, and the sixteen upstream attributes of its pixels."""

import math
import os

import numpy as np

from . import raster
from .boundary import BOUNDARY_NAMES, boundary_walk
from .earth import model_code
from .grid import pixel_edges_deg, snap_transform
from .network import check_index, dfs_sequence, downstream_index
from .shape import SHAPE_NAMES, shape_arithmetic
from .slope import local_slope
from .terrain import TERRAIN_NAMES, upstream_terrain

NODATA = -9999.0
MIN_UPSTREAM_AREA_KM2 = 10.0

# the sixteen in the order of Table 3 of the FullBasin paper
ATTRIBUTE_NAMES = (
    "mean_elv", "min_elv", "max_elv", "elv_std", "mean_slp", "slp_std",
    "perimeter_km", "basin_length_km", "basin_width_km", "convexity",
    "relief", "hypsometric_integral", "elongation_ratio", "gravelius", "circularity", "lemniscate_ratio",
)


def _frozen(array):
    """A cached or kept array made read-only: it is handed out on every call, and a caller who wrote to it would
    change every later result with no sign.  Take a ``.copy()`` to modify one."""
    array.flags.writeable = False
    return array


def _kept(values, dtype):
    """A read-only copy of an input, so that the caller's own array is neither changed nor frozen."""
    return _frozen(np.array(values, dtype=dtype, copy=True, order="C"))


def _check_longitude_latitude(transform, shape, crs):
    """The grid must lie in longitude and latitude: its CRS geographic when one is given, and its extent inside
    [-540, 540] degrees of longitude and [-90, 90] of latitude (a projected grid in metres is far outside)."""
    if crs is not None:
        from rasterio.crs import CRS
        if not CRS.from_user_input(crs).is_geographic:
            raise ValueError(f"FlowMorph reads longitude-latitude grids in degrees; the CRS given is {crs}")
    nrow, ncol = shape
    west, north = transform[0], transform[3]
    east = west + ncol * transform[1]
    south = north + nrow * transform[5]
    if not (-540.0 <= west and east <= 540.0 and -90.0 - 1e-9 <= south and north <= 90.0 + 1e-9):
        raise ValueError(f"the grid spans longitude {west:g} .. {east:g} and latitude {south:g} .. {north:g}; "
                         f"FlowMorph reads longitude-latitude grids in degrees")


class FlowMorph:
    """The upstream attributes of every pixel of a D8 grid that holds whole basins.

    The inputs are copied, and every array the object keeps or hands out is read-only; take a ``.copy()`` to
    change one.

    Parameters
    ----------
    d8 : 2-D uint8 array, MERIT Hydro's flow directions (247 no data, 0 and 255 pits)
    upa : 2-D float array, the upstream area in km^2 (MERIT Hydro's own ``upa``): it chooses the targets and is the
        A of the convexity and of step 4; it also orders the donors of a confluence when ``seq_dfs`` is not given
    transform : the six numbers of the grid (GDAL order), longitude and latitude in degrees, north up; its decimal
        noise is removed first (:func:`flowmorph.grid.snap_transform`)
    elevation : 2-D float array, -9999 where there is none; needed for steps 1, 2 and 4
    ring : optional (nrow + 2) x (ncol + 2) elevation, the grid with one more cell on every side, for the slope at
        the grid's edge (see :func:`flowmorph.slope.local_slope`)
    crs : kept for writing
    seq_dfs, upg : optional index (FlowTopo's ``seq_dfs`` and the upstream pixel count); checked against the flow
        directions when given, made from them when not
    earth : ``"wgs84"`` (default) or ``"sphere"``, the earth of the lengths and hull areas of step 3
    min_upstream_area_km2 : the targets are the pixels whose upstream area is finite and at least this (10)
    """

    def __init__(self, d8, upa, transform, elevation=None, ring=None, crs=None, seq_dfs=None, upg=None,
                 earth="wgs84", min_upstream_area_km2=MIN_UPSTREAM_AREA_KM2):
        d8 = np.asarray(d8)
        if d8.ndim != 2:
            raise ValueError("the flow directions must be a 2-D grid")
        self.shape = d8.shape
        transform = tuple(float(value) for value in transform)
        if len(transform) != 6 or not all(math.isfinite(value) for value in transform):
            raise ValueError("the transform must be six finite numbers")
        if transform[2] != 0.0 or transform[4] != 0.0 or not transform[1] > 0.0 or not transform[5] < 0.0:
            raise ValueError("the grid must be north up with no rotation")
        self.transform = snap_transform(transform)
        self._pixel_edges = pixel_edges_deg(transform)
        _check_longitude_latitude(self.transform, self.shape, crs)
        self.crs = crs
        self.earth = earth
        self._model = model_code(earth)
        self.min_upstream_area_km2 = float(min_upstream_area_km2)
        if not (math.isfinite(self.min_upstream_area_km2) and self.min_upstream_area_km2 > 0.0):
            raise ValueError("min_upstream_area_km2 must be a finite area above 0 km2")
        self.idxs_ds = _frozen(downstream_index(d8))
        self.upa = _kept(upa, np.float32)
        if self.upa.shape != self.shape:
            raise ValueError("upa must have the shape of the flow directions")
        network = self.idxs_ds >= 0
        with np.errstate(invalid="ignore"):
            self.is_target = _frozen((np.isfinite(self.upa) & (self.upa >= np.float32(self.min_upstream_area_km2))
                                      ).reshape(-1) & network)
        if seq_dfs is None and upg is None:
            seq_dfs, upg = dfs_sequence(self.idxs_ds, self.upa)
            self.seq_dfs, self.upg = _frozen(seq_dfs), _frozen(upg)
        elif seq_dfs is not None and upg is not None:
            check_index(seq_dfs, upg, self.idxs_ds)
            self.seq_dfs = _kept(np.asarray(seq_dfs).reshape(-1), np.int32)
            self.upg = _kept(np.asarray(upg).reshape(-1), np.int64)
        else:
            raise ValueError("give seq_dfs and upg together, or neither")
        self.elevation = None if elevation is None else _kept(elevation, np.float32)
        if self.elevation is not None and self.elevation.shape != self.shape:
            raise ValueError("the elevation must have the shape of the flow directions")
        self.ring = None if ring is None else _kept(ring, np.float32)
        self._cache = {}

    @classmethod
    def from_raster(cls, dir_path, upa_path, elv_path=None, ring_path=None, seq_dfs_path=None, upg_path=None,
                    **kwargs):
        """Read the grids from GeoTIFF files; ``ring_path`` is a larger elevation (the global one) the ring of one
        cell round the grid is read from."""
        d8, transform, crs = raster.read_d8(dir_path)
        upa = raster.read_float(upa_path, d8.shape, transform)
        elevation = None if elv_path is None else raster.read_float(elv_path, d8.shape, transform)
        ring = None if ring_path is None else raster.read_ring(ring_path, d8.shape, transform)
        seq_dfs = None if seq_dfs_path is None else raster.read_int(seq_dfs_path, d8.shape, transform)
        upg = None if upg_path is None else raster.read_int(upg_path, d8.shape, transform)
        if upg is not None:
            upg[upg < 0] = 0
        return cls(d8, upa, transform, elevation=elevation, ring=ring, crs=crs, seq_dfs=seq_dfs, upg=upg, **kwargs)

    # ---- the four steps ----

    def local_slope(self):
        """Step 1: Horn's slope in degrees, float32, -9999 where there is no elevation."""
        if "slp" not in self._cache:
            if self.elevation is None:
                raise ValueError("the slope needs the elevation")
            self._cache["slp"] = _frozen(local_slope(self.elevation, self.transform, self.ring))
        return self._cache["slp"]

    def upstream_terrain(self):
        """Step 2: mean_elv, min_elv, max_elv, elv_std, mean_slp, slp_std."""
        if "terrain" not in self._cache:
            if self.elevation is None:
                raise ValueError("the terrain attributes need the elevation")
            values = upstream_terrain(self.seq_dfs, self.upg, self.idxs_ds, self.elevation, self.local_slope(),
                                      self.is_target)
            self._cache["terrain"] = {name: _frozen(values[name].reshape(self.shape)) for name in TERRAIN_NAMES}
        return dict(self._cache["terrain"])      # a new dict: replacing an entry leaves the cache alone

    def boundary_walk(self):
        """Step 3: perimeter_km, basin_length_km, basin_width_km, convexity."""
        if "boundary" not in self._cache:
            values = boundary_walk(self.seq_dfs, self.upg, self.upa, self.is_target, self.shape, self.transform,
                                   self._model, self._pixel_edges)
            self._cache["boundary"] = {name: _frozen(values[name].reshape(self.shape)) for name in BOUNDARY_NAMES}
        return dict(self._cache["boundary"])      # a new dict: replacing an entry leaves the cache alone

    def shape_arithmetic(self):
        """Step 4: relief, hypsometric_integral, elongation_ratio, gravelius, circularity, lemniscate_ratio."""
        if "shape" not in self._cache:
            terrain = self.upstream_terrain()
            boundary = self.boundary_walk()
            values = shape_arithmetic(
                self.upa, boundary["perimeter_km"], boundary["basin_length_km"], terrain["mean_elv"],
                terrain["min_elv"], terrain["max_elv"], self.is_target.reshape(self.shape))
            self._cache["shape"] = {name: _frozen(grid) for name, grid in values.items()}
        return dict(self._cache["shape"])      # a new dict: replacing an entry leaves the cache alone

    def attributes(self):
        """The slope and the sixteen attributes, name -> float32 grid (-9999 where not written)."""
        result = {"slp": self.local_slope()}
        result.update(self.upstream_terrain())
        result.update(self.boundary_walk())
        result.update(self.shape_arithmetic())
        return {name: result[name] for name in ("slp",) + ATTRIBUTE_NAMES}

    def write(self, directory, names=None, suffix=""):
        """Write attributes as float32 GeoTIFFs ``<name><suffix>.tif`` into a directory; all of them by default.
        Returns the paths written."""
        os.makedirs(directory, exist_ok=True)
        values = self.attributes() if names is None else {name: self._one(name) for name in names}
        paths = []
        for name, grid in values.items():
            path = os.path.join(directory, f"{name}{suffix}.tif")
            raster.write_float(path, grid, self.transform, self.crs)
            paths.append(path)
        return paths

    def _one(self, name):
        if name == "slp":
            return self.local_slope()
        if name in TERRAIN_NAMES:
            return self.upstream_terrain()[name]
        if name in BOUNDARY_NAMES:
            return self.boundary_walk()[name]
        if name in SHAPE_NAMES:
            return self.shape_arithmetic()[name]
        raise ValueError(f"unknown attribute {name!r}; the names are {('slp',) + ATTRIBUTE_NAMES}")


__all__ = ["FlowMorph", "ATTRIBUTE_NAMES", "MIN_UPSTREAM_AREA_KM2"]
