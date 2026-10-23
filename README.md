# FlowMorph

The shape and terrain of every pixel's upstream catchment on a D8 flow-direction grid.

For every pixel whose upstream area is at least 10 km², FlowMorph describes the catchment that drains to that
pixel with sixteen attributes: its elevation and slope, the plan form of its boundary, and the shape ratios made
from those. These are the sixteen per-pixel upstream attributes of the FullBasin dataset (Table 3 of the FullBasin
paper, Jiang et al., under review at Earth System Science Data). This repository is the Python version of the C
code of FlowMorph 1.1, which computed the FullMorph v1.1 layers of MERIT Hydro, and gives the same numbers on the
same grid (see "Same numbers as the C code").

The sixteen layers for the whole of MERIT Hydro (FullMorph v1.1):
<https://fullhydro.org/downloads/merit-hydro-90m/fullbasin/fullmorph/v1.1/>

Documentation and the other FullHydro tools: <https://fullhydro.org/tools/>

Author: Lulu Jiang (<https://lulujiang.me>)

## Install

```sh
git clone https://github.com/hellolulujiang/flowmorph.git
cd flowmorph
pip install -e ".[test]"
pytest
```

Python 3.10 or later with `numpy`, `numba`, `scipy` and `rasterio`.

## Use

```python
import flowmorph

fm = flowmorph.FlowMorph.from_raster("data/dir_example.tif", "data/upa_example.tif", "data/elv_example.tif")
attributes = fm.attributes()          # name -> float32 grid, -9999 where not written; read-only, .copy() to edit
fm.write("out")                       # seventeen GeoTIFFs: slp and the sixteen attributes
```

or from the command line:

```sh
python -m flowmorph data/dir_example.tif data/upa_example.tif data/elv_example.tif out
```

The bundled example is one basin of MERIT Hydro in Tasmania (696 km², 110,125 pixels, 2,033 of them with 10 km²
or more upstream).

## Input

Three grids of the same size, north up, in longitude and latitude (degrees). A pixel holds no data where it holds
247 (the directions), -9999 or a value that is not finite (the float grids) or -1 (an index). A float grid or an
index that declares another no-data value has it taken as no data too; a direction grid must declare 247 or
nothing. The C programs know the first three only, and the C terrain step refuses a float grid that does not
declare -9999: on MERIT Hydro's files the two agree; on a file that declares another value this package takes the
file at its word. A mask band or an alpha band that hides pixels holding values is refused, as the C programs would read
those values.

* `dir`: flow directions in the MERIT Hydro convention, Byte: a power of two clockwise from east (1 E, 2 SE, 4 S,
  8 SW, 16 W, 32 NW, 64 N, 128 NE), 0 a river mouth and 255 an inland depression (both end a basin), 247 no data.
  The grid must hold whole basins: a pixel that flows out of the grid or into a pixel without flow direction is
  refused, and so is a cycle.
* `upa`: the upstream area in km², MERIT Hydro's own. It chooses the pixels that get the attributes (finite and at
  least 10 km²; `min_upstream_area_km2` sets another threshold), it is the A of the ratios, and it orders the
  donors at a confluence.
* `elv`: the elevation in metres, -9999 where there is none. Every pixel of the network needs one.

The slope of a pixel on the edge of the grid needs its neighbours outside the grid. Give `ring_path` (a larger
elevation grid, typically the global one the grid was cut from) and they are read from there; without it they
take the centre value.

## The sixteen attributes

Written where the upstream area is at least 10 km², -9999 elsewhere. A is the upstream area, P the perimeter, L_b the
basin length and W_b the basin width.

| # | name | unit | what |
|---|---|---|---|
| 1 | `mean_elv` | m | mean elevation of the catchment, weighted by pixel area |
| 2 | `min_elv` | m | lowest elevation |
| 3 | `max_elv` | m | highest elevation |
| 4 | `elv_std` | m | standard deviation of elevation, weighted by pixel area (population) |
| 5 | `mean_slp` | deg | mean local slope (Horn's 3 x 3 kernel), weighted by pixel area |
| 6 | `slp_std` | deg | standard deviation of the local slope, weighted by pixel area (population) |
| 7 | `perimeter_km` | km | P: FlowMorph 1.0's perimeter, the Moore trace through the boundary pixels' centres with the step weights of Vossepoel & Smeulders (1982) for square pixels, taken on cells close to square on the ground: k columns merged into a cell, k the pixel's height over its width at the catchment's middle, interpolated between the two whole numbers next to it |
| 8 | `basin_length_km` | km | L_b: length of the great circle from the pixel to the farthest boundary pixel (the basin diameter of Rigon et al., 1996) |
| 9 | `basin_width_km` | km | W_b: span of the boundary across the great circle of L_b: the farthest boundary pixel on either side, each measured to the foot of its perpendicular (Rigon et al., 1996; Willemin, 2000); where two boundary pixels are nearly equally far from the pixel, the axis, and with it W_b, can change abruptly from one pixel to the next |
| 10 | `convexity` | - | A over the area of the convex hull of the catchment's pixels on the sphere (its area on the ellipsoid), at most 1 (Willemin, 2000) |
| 11 | `relief` | m | max_elv - min_elv (Strahler, 1952) |
| 12 | `hypsometric_integral` | - | HI = (mean_elv - min_elv) / relief (Strahler, 1952); no data when the relief is 0 |
| 13 | `elongation_ratio` | - | R_e = (2 / L_b) sqrt(A / pi) (Schumm's formula, with L_b) |
| 14 | `gravelius` | - | Gc = P / (2 sqrt(pi A)), the Gravelius compactness (Gravelius, 1914) |
| 15 | `circularity` | - | R_c = 4 pi A / P², the circularity ratio (Miller, 1953) |
| 16 | `lemniscate_ratio` | - | perimeter of the lemniscate with area A and length L_b, over P (Chorley et al., 1957) |

The local slope itself, `slp`, is written as well. The docstrings of `flowmorph/terrain.py`,
`flowmorph/boundary.py`, `flowmorph/vscells.py`, `flowmorph/sphere.py` and `flowmorph/shape.py` give each definition in
full: which pixels are on the boundary, how the contour is traced and on which cells, which pixel is the farthest,
how the hull is drawn and how its area is taken, and the two branches of the lemniscate.

The basin length, the width and the hull (attributes 8 to 10) are measured on the sphere of the geodetic latitudes
and longitudes: the basin length, the two legs of the width and the sides of the hull are great circles. Their
lengths and the hull's area are taken on the WGS84 ellipsoid, a length as the sum of pieces of at most 5 km; a step
of the perimeter takes the edges of a pixel at its latitude on the same ellipsoid. `earth="sphere"`
(`--earth sphere`) takes them all on the authalic sphere of radius 6,371,007.181 m instead.

The weights of the perimeter were made for square pixels. On a grid with the same angular step in longitude and
latitude (MERIT Hydro, HydroSHEDS) a pixel is about 1 / cos(latitude) times as tall as it is wide on the ground (2 at
60°, 3.9 at 75°, 5.8 at 80°), and on such pixels 1.0's perimeter reads long: on
smooth shapes of known perimeter 2 % at 60° and 10 % at 80°. On cells close to square it reads as it does at the
equator, at every latitude: 0.98 of the perimeter for radii of 10 to 25 pixels and 0.994 for 30 to 80. The merge
measures on cells close to square on the ground, which fills the notches of a divide narrower than a cell; where
the k cells start (at the global columns that are multiples of k) moves the perimeter of a catchment by a median
0.4 to 1.1 %, with no bias. Where a pixel is wider than tall, within 6.59° of the equator, the perimeter is 1.0's
on the pixels themselves.

## How it works

The pixels of each basin are numbered depth first from the outlet, the larger upstream area first at every
confluence, and the numbering is turned so that the outlet comes last (FlowTopo's `seq_dfs`). The catchment of a
pixel p, p included, is then the run of ranks `[seq_dfs(p) - upg(p) + 1, seq_dfs(p)]`, where `upg` is its number of
upstream pixels, and "is q upstream of p" is two comparisons.

* Terrain: the ranks are taken from 0 up. When the pixel of rank r is reached, the sums of the catchments flowing
  into it are on the top of a stack; they are taken off and added to r's own values, and r's sums go back on the
  stack. One pass, no per-pixel accumulators.
* Plan form: the pixels with 10 km² or more upstream are taken upstream first. A run of them each of whose
  catchment holds the one before is walked with one set of boundary pixels, grown from one catchment to the next:
  only the pixels new to the catchment are tested, and only the old boundary pixels next to them are tested again.
  For the perimeter the outer contour is traced through the boundary pixels' centres (Moore neighbourhood, Jacob's
  stopping rule) on the merged cells, twice where the merge is taken between two whole numbers of columns.

A catchment whose convex hull reaches more than 80 degrees (some 8,900 km) from its pixel, or rings a pole, is
refused with a `ValueError`: a limit of the method, which no river basin on Earth comes near.

`seq_dfs` and `upg` can be given (`seq_dfs=`, `upg=`, for example FlowTopo's layers); they are then checked
against the flow directions before they are used.

## Same numbers as the C code

The FullMorph v1.1 layers were computed with the C code of FlowMorph 1.1, region by region on FullBasin's partition of
MERIT Hydro, from MERIT Hydro's flow directions and upstream area, MERIT DEM's elevation and FlowTopo's `seq_dfs`.
Given a region's inputs as the C code read them (`seq_dfs=` and `upg=`, and the slope's ring read from the global
elevation, `ring_path=` or `--ring`), this package writes the same seventeen layers, bit for bit. Checked on two
regions:

* region 5699 (Tasmania, 39 to 44° S; 6,000 by 7,200 pixels, 10,715,913 of them on the network, 178,129 targets),
  where the perimeter is interpolated between the pixels and cells of two columns: about 35 s on one thread and
  8 GB of memory (the largest resident set of the process);
* region 10001 (64 to 72° N, across the 180th meridian; 9,600 by 9,600 pixels, 19,168,845 on the network, 286,915
  targets), where it is taken on cells of two to four columns: about 110 s.

Without the ring, a pixel on the edge of the grid takes the centre value for its missing neighbours, and its slope,
and the mean_slp and slp_std of the catchments that hold it, differ from the C's.

The perimeter is also checked apart from the regions: on the 2,033 catchments of the bundled basin (42° S, cells of
one and of two columns) and on 23 made-up catchments at 62 to 83.5° (cells of two to nine columns, two grids west of
180 W, three catchments that touch the grid's edges), the package and the C code (its program
`flowmorph_vs_cells_dump`, given the same catchments) give the same doubles; the tests hold both
(`tests/reference/expected_c_perimeter.npz`, `tests/reference/expected_c_vs_cells_synthetic.npz`, and the scripts
that made them beside them).

The C binary on macOS takes the sine and the cosine of one angle from one library routine, whose last bit can
differ from the two separate routines; on macOS the package calls the same routine (`flowmorph/clib.py`), elsewhere
the two separate ones, as a C compiled there does.

To place the pixels the same way, the transform of a grid is taken as the C code takes it
(`flowmorph.grid.snap_transform`): the pixel size back on 1/N of MERIT Hydro's or HydroSHEDS' grid, the origin
back on the grid of half pixels, every term rounded to 1e-12 degree. The files written carry the exact grid, the
pixel size and the origin without that rounding (`flowmorph.grid.exact_transform`: 1/1200 of a degree on MERIT
Hydro's grid, not 0.000833333333). A window cut out of a larger grid has another
origin, and the last bits of some pixel centres move with it: on the bundled basin, cut from region 5699, 16 slopes
and 2 attributes (one mean slope, one convexity) differ from the C's region tile in their last place.

The bundled elevation is MERIT Hydro's, and the reference of the tests (`tests/reference/expected_c.npz`) holds the
values an earlier state of the C code wrote on it. The tests compare the example with them to a relative 5e-7, but
for two groups of layers that the C code has changed since. The five layers weighted by pixel area (`mean_elv`,
`elv_std`, `mean_slp`, `slp_std`, `hypsometric_integral`; the reference counts every pixel once) are checked against
a plain recomputation of the weighted sums. The perimeter is checked bit for bit against the C code's perimeter of
every target of the window (`tests/reference/expected_c_perimeter.npz`, made by
`tests/reference/make_expected_c_perimeter.py`), and the Gravelius compactness, the circularity and the lemniscate
ratio against the reference scaled by the change of the perimeter.

## Changes from FlowMorph 1.0

1.1 computes the same sixteen attributes as FlowMorph 1.0, with these changes:

* the perimeter is 1.0's, taken on cells close to square on the ground (1.0 took it on the pixels, for which its
  weights, made for square pixels, read long where a pixel is taller than wide: on smooth shapes of known perimeter
  2 % at 60° and 10 % at 80°). On the bundled basin (42° S) it is a median 0.4 % shorter than 1.0's trace on the
  pixels taken with the other definitions of this version (2.0 % shorter to 1.3 % longer);
* the basin width and the convex hull are measured on the sphere (1.0: in one plane centred on the outlet, where
  the width of a large river far from the equator comes out up to some 30 % short), and the basin length is the
  great circle to the boundary pixel farthest by angle (1.0: the distance to the boundary pixel farthest by that
  distance). On the 2,033 catchments of the bundled basin the sphere and that plane differ in the width by less
  than 0.2 % for 98 in 100 and in the hull by less than 0.03 %; at 25 of them another boundary pixel is the
  farthest, the length differs by up to 0.2 % and the width, across a turned axis, by -19 to +12 %;
* the means and standard deviations of elevation and slope are weighted by pixel area, and the hypsometric
  integral follows the mean (1.0: every pixel counted once); a network pixel without a slope is refused (1.0: it
  entered the mean slope as 0 degrees).

The FullBasin v1.0 layers on Zenodo were computed with code that came before FlowMorph 1.0. Besides the changes
above, FlowMorph 1.0 already differed from them:

* the convex hull of the convexity is that of the pixels, not of their centres; the centres' hull is half a pixel
  short all round, so a small catchment came out too convex;
* lengths, edges and hull areas are on the WGS84 ellipsoid (FullBasin v1.0: a sphere); `earth="sphere"` measures on
  that sphere, with the definitions of this version;
* the area A is MERIT Hydro's own upstream area (FullBasin v1.0: the upstream area recomputed on the sphere of
  radius 6,371 km), which approximates the ellipsoid (CaMa-Flood's formula): it is below
  the exact area of the same pixels by 0 at the equator, 0.14 % at 40 degrees and 0.26 % at 60, and a ratio of A to
  a length or an area measured on the ellipsoid (the convexity, the elongation ratio, ...) carries that offset;
* the boundary trace stops when it is back at its start and its next step would repeat its first (FullBasin v1.0
  stopped at the first return, and left out a part of the catchment joined to the rest only through the start
  pixel);
* the width is taken in one plane centred on the outlet (FullBasin v1.0: three latitudes mixed in one width);
* `lemniscate_ratio` is Chorley's ratio of perimeters (FullBasin v1.0 stores the parameter k under that name,
  which equals 1 / elongation_ratio²);
* the slope of a pixel at the edge of a region reads its neighbours from the global elevation.

## Tests

```sh
pytest
```

`tests/test_example.py` runs the bundled basin and compares every attribute with the C code's (the five weighted
by area with a recomputation, see above); `tests/test_contracts.py`
checks the rules and the refusals on small grids made up for the purpose; `tests/test_plan_form.py` checks the
perimeter, the length, the width and the hull on shapes whose values are known beforehand, and the perimeter of
made-up catchments against the C code's; `tests/test_raster_validation.py` checks what is refused of a raster (another reference system, a grid not in degrees
or not north up, a mask over pixels that hold values) and that distances stay finite at the edge of their domain;
`tests/test_threshold_bounds.py` checks which area thresholds are taken.

## Citing

The attributes are described in the FullBasin paper (Jiang et al., under review at Earth System Science Data). Cite
that paper, and this repository with the release you used (1.1.0).

## Acknowledgements

**MERIT Hydro** (Yamazaki et al., 2019; [10.1029/2019WR024873](https://doi.org/10.1029/2019WR024873)) is the grid
FlowMorph was built on, and the source of the bundled example.

## Contact

<lulu_jiang@pku.edu.cn>, or an issue on this repository.

## Licence

MIT for the code; see [`LICENSE`](LICENSE). The bundled MERIT Hydro excerpt keeps its own terms; see
[`DATA_NOTICE.md`](DATA_NOTICE.md).
