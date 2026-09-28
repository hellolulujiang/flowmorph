"""FlowMorph -- the shape and terrain of every pixel's upstream catchment on a D8 grid.

For every pixel whose upstream area is at least 10 km^2, FlowMorph describes the catchment that drains to it with
sixteen attributes (Table 3 of the FullBasin paper):

* terrain, summed over the catchment's pixels: mean, lowest and highest elevation, the standard deviation of
  elevation, and the mean and standard deviation of the local slope (Horn 1981);
* plan form, from the pixels on the catchment's boundary: the perimeter (Vossepoel & Smeulders 1982), the basin
  length and width (Rigon et al. 1996), and the convexity (Willemin 2000);
* shape, from those: relief, hypsometric integral, elongation ratio, Gravelius coefficient, circularity and the
  lemniscate ratio of Chorley et al. (1957).

The catchment of a pixel is a run of FlowTopo's depth-first ranks, [seq_dfs - upg + 1, seq_dfs], so the terrain
attributes are one pass over the ranks and the boundary of each catchment is grown from the one upstream of it.

Quick start
-----------
>>> import flowmorph                                                     # doctest: +SKIP
>>> fm = flowmorph.FlowMorph.from_raster("dir.tif", "upa.tif", "elv.tif")  # doctest: +SKIP
>>> fm.write("out")                                                       # doctest: +SKIP
"""

from .api import ATTRIBUTE_NAMES, MIN_UPSTREAM_AREA_KM2, FlowMorph
from .boundary import BOUNDARY_NAMES, boundary_walk
from .earth import EARTH_MODELS
from .network import check_index, dfs_sequence, downstream_index
from .shape import SHAPE_NAMES, shape_arithmetic
from .slope import local_slope
from .terrain import TERRAIN_NAMES, upstream_terrain

__version__ = "1.0.0"

__all__ = [
    "FlowMorph", "ATTRIBUTE_NAMES", "MIN_UPSTREAM_AREA_KM2", "EARTH_MODELS",
    "downstream_index", "dfs_sequence", "check_index",
    "local_slope", "upstream_terrain", "boundary_walk", "shape_arithmetic",
    "TERRAIN_NAMES", "BOUNDARY_NAMES", "SHAPE_NAMES",
]
