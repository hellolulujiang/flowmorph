# Changelog

## 1.1.0 (2026-10-23)

The Python version of FlowMorph 1.1, the C code that computed the FullMorph v1.1 layers of MERIT Hydro: on regions
5699 and 10001 of FullBasin's partition the seventeen layers equal the C code's bit for bit (README, "Same numbers as
the C code"). The README lists the changes from FlowMorph 1.0 with their sizes; in short:

Results that change:

* **The perimeter is 1.0's, taken on cells close to square on the ground** (`perimeter_km`, `flowmorph/vscells.py`).
  The Moore trace through the boundary pixels' centres with the step weights of Vossepoel & Smeulders (1982), as in
  1.0, on cells of k columns: a cell is in the catchment when one of its k pixels is, its east-west edge is k pixel
  widths, and k is the pixel's height over its width at the middle of the catchment's rows, the perimeter
  interpolated between the two whole numbers next to it, so that it has no step from one latitude to the next. The
  weights were made for square pixels: on the pixels of a longitude-latitude grid 1.0's perimeter read 2 % long at
  60° and 10 % at 80° on smooth shapes of known perimeter; on the cells it reads as at the equator at every
  latitude. Where a pixel is wider than tall (within 6.59° of the equator) the perimeter is 1.0's on the pixels.
  The cells start at the global columns that are multiples of k (`vscells.global_col_of_col0`), so a catchment has
  the same perimeter in any window of MERIT Hydro; beyond 89.5° of latitude the merge is held at that of 89.5°. A
  catchment that the merge leaves one or two cells, or that has fewer than three boundary pixels, has no perimeter.
  The Gravelius compactness, the circularity and the lemniscate ratio follow the perimeter.
* **The basin width and the convex hull are measured on the sphere** (`flowmorph/sphere.py`): the width across the
  great circle of the basin length, and the hull of the pixels with great-circle sides, its area on the earth model.
  1.0 took them in one plane centred on the pixel, where the width of a large catchment far from the equator came
  out up to some 30 % short. The basin length is the great circle to the boundary pixel farthest by angle (1.0: the
  boundary pixel farthest by distance).
* **The means and standard deviations of elevation and slope are weighted by pixel area** (`mean_elv`, `elv_std`,
  `mean_slp`, `slp_std`; the hypsometric integral follows the mean). Every pixel counts with its area on the WGS84
  ellipsoid (`FlowMorph.pixel_area()`; `upstream_terrain` takes the areas as an argument). 1.0 counted every pixel
  once, and a catchment that spans many degrees of latitude came out too low or too high: at the mouth of the
  Yenisei 659.6 m where the mean over the catchment's area is 726.1 m. Of the basins of MERIT Hydro of 10 km² and
  more, 98.7 % change by less than 1 m.
* **A network pixel without a slope is refused** (`upstream_terrain`), as one without an elevation is; 1.0 let a slope
  of -9999 into the mean slope as 0 degrees.

Inputs and outputs (no number changes):

* The grids must be in longitude and latitude (degrees), north up, with matching reference systems (the elevation
  ring and the integer indices included); all six terms of the transforms are compared, and two headers that write
  one grid with different rounding of the pixel size are one grid. `local_slope` and `boundary_walk`, called on
  their own, refuse a rotated or sheared grid, as the `FlowMorph` object does.
* A pixel that holds the file's declared no-data value is no data, whatever a mask says; a mask band or an alpha band
  that hides pixels holding values is refused, as the C programs would read those values.
* Refused with a message: a target threshold that is not finite or not above 0 as float32, an infinity or a NaN at a
  target in `shape_arithmetic`, a grid of 2^31 pixels or more, a catchment whose contour does not close or whose hull
  cannot be measured (`boundary_walk`).
* The files written carry the exact grid (`flowmorph.grid.exact_transform`: 1/1200 of a degree on MERIT Hydro), not
  the transform rounded to 1e-12 that the computations take; a file that needs it is a BigTIFF; `write` refuses a
  name that is taken (`FileExistsError`) and keeps the order of a list or a tuple of names.
* Every array a `FlowMorph` object keeps or hands out is read-only; take a `.copy()` to change one.
* Distances stay finite between two points opposite each other on the Earth.

New: `flowmorph/vscells.py`, `flowmorph/sphere.py`, `flowmorph/clib.py` (three routines of the C library, called as
the C code calls them); `tests/test_plan_form.py`, `tests/test_raster_validation.py`, `tests/test_threshold_bounds.py`.
`tests/reference/expected_c_perimeter.npz` and `tests/reference/expected_c_vs_cells_synthetic.npz` hold the C code's
perimeter of the bundled basin and of 23 made-up catchments at 62 to 83.5°, with the scripts that made them.

## 1.0.0 (2026-09-28)

First release: the local slope and the sixteen upstream attributes of every pixel with 10 km² or more upstream, on a
MERIT Hydro D8 grid of whole basins. The differences from the FullBasin v1.0 layers are listed in the
[user guide of 1.0.0](https://github.com/hellolulujiang/flowmorph/blob/v1.0.0/docs/user-guide.md#differences-from-the-fullbasin-v10-layers).
