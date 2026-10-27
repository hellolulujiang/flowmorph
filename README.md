# FlowMorph

The shape and terrain of every pixel's upstream catchment on a D8 flow-direction grid.

FlowMorph computes sixteen attributes for every pixel whose upstream area is at least 10 km². They describe the
catchment's elevation and slope, the plan form of its boundary, and the shape ratios made from those, as defined in
the FullBasin paper (Jiang et al., under review at Earth System Science Data).

The sixteen attributes for the whole of MERIT Hydro are in
[FullMorph v1.1](https://fullhydro.org/downloads/merit-hydro-v1.0.1-90m/fullbasin/fullmorph/v1.1/).

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

`python -m flowmorph --help` lists the options: `--min-area` (the targets' smallest upstream area, km², default 10),
`--earth sphere`, `--ring` (a larger elevation grid for the slope at the grid's edge) and `--seq-dfs` with `--upg`
(FlowTopo's sequence of the grid and its upstream pixel count, made from the flow directions when not given).

The bundled example is one basin of MERIT Hydro in Tasmania (696 km², 110,125 pixels, 2,033 of them with 10 km²
or more upstream).

## Input

Three grids of the same size, north up, in longitude and latitude (degrees). A pixel holds no data where it holds
247 (the directions), -9999 or a value that is not finite (the float grids) or -1 (an index). A float grid or an
index that declares another no-data value has it taken as no data too; a direction grid must declare 247 or
nothing. A mask band or an alpha band that hides pixels holding values is refused.

A header's pixel size and origin are put back on the exact grid when they differ from it only by rounding (by less
than 1e-5 of the size, 1e-4 of a half pixel): the size on 1/N of a degree, N a divisor of 1200 or 3600, the origin
on the grid of half pixels. The files written carry that exact grid (`flowmorph/grid.py`).

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

FullMorph v1.1 was computed with this method region by region on the 90 regions of
[FullDivide v1.1](https://fullhydro.org/downloads/merit-hydro-v1.0.1-90m/fullbasin/fulldivide/v1.1/), with the
elevation of MERIT DEM v1.0.3 and, for the slope at a region's edge, the terrain beyond it (`ring_path` above). The
bundled example's `elv` is MERIT Hydro's own elevation, so its terrain values are not those of FullMorph.

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
| 7 | `perimeter_km` | km | P: the Moore trace through the boundary pixels' centres with the step weights of Vossepoel & Smeulders (1982), on cells close to square on the ground (see below and `flowmorph/vscells.py`) |
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
latitude (MERIT Hydro, HydroSHEDS) a pixel is about 1 / cos(latitude) times as tall as it is wide on the ground (2
at 60°, 3.9 at 75°, 5.8 at 80°), and the trace with these weights on the pixels themselves reads long: on smooth
shapes of known perimeter 2 % at 60° and 10 % at 80°. On cells close to square (k columns merged into a cell, k the
pixel's height over its width at the catchment's middle, interpolated between the two whole numbers next to it) it
reads as the same trace does at the equator, at every latitude: 0.98 of the perimeter for radii of 10 to 25 pixels
and 0.994 for 30 to 80. The merge measures on cells close to square on the ground, which fills the notches of a
divide narrower than a cell; where the k cells start (at the global columns that are multiples of k) moves the
perimeter of a catchment by a median 0.4 to 1.1 %, with no bias. Where a pixel is wider than tall, within 6.59° of
the equator, the perimeter is the weighted trace on the pixels themselves.

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

## Tests

```sh
pytest
```

`tests/test_example.py` runs the bundled basin against reference values in `tests/reference/expected.npz` and
`tests/reference/expected_perimeter.npz`, with a recomputation of the attributes weighted by pixel area.

* `tests/test_contracts.py` checks rules, refusals and formulas on small made-up grids.
* `tests/test_plan_form.py` checks the perimeter, the length, the width and the hull on shapes of known values, and
  the perimeter of made-up catchments against reference values.
* `tests/test_raster_validation.py` checks spatial alignment, no-data masks and distances at their domain's edge.
* `tests/test_threshold_bounds.py` checks which area thresholds are taken.

## Citing

The attributes are described in the FullBasin paper (Jiang et al., under review at Earth System Science Data). Cite
that paper, and this repository with the release you used.

## Acknowledgements

**MERIT Hydro** (Yamazaki et al., 2019; [10.1029/2019WR024873](https://doi.org/10.1029/2019WR024873)) is the grid
FlowMorph was built on, and the source of the bundled example.

## Contact

<lulu_jiang@pku.edu.cn>, or an issue on this repository.

## Licence

MIT for the code; see [`LICENSE`](LICENSE). The bundled MERIT Hydro excerpt keeps its own terms; see
[`DATA_NOTICE.md`](DATA_NOTICE.md).
