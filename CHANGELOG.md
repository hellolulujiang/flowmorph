# Changelog

## 1.1.1 (2026-10-28)

* The bundled MERIT Hydro excerpt keeps both licences, CC BY-NC 4.0 or ODbL 1.0 at the user's choice.
  `DATA_NOTICE.md` names both options and the source version, MERIT Hydro v1.0.1.
* Documentation and comments describe the methods themselves. The FullMorph download link is fixed.
* The perimeter test in metres compares to a relative 1e-12. It failed on Linux, where the system's `sin` and `cos`
  can differ from those of macOS in their last bits; the float32 kilometres are still compared bit for bit.
* The reference files in `tests/reference/` are renamed (`expected.npz`, `expected_perimeter.npz`,
  `expected_perimeter_synthetic.npz`), and the two scripts that made the perimeter references are removed. Tests are
  renamed for the reference values they compare with.
* Results are unchanged.

## 1.1.0 (2026-10-23)

* The perimeter uses the Moore trace with the step weights of Vossepoel & Smeulders (1982) on cells close to square
  on the ground. It is interpolated between merges of two neighbouring whole numbers of columns.
* Basin length, width and the convex hull are measured on the sphere, with lengths and areas on the chosen earth
  model. The farthest boundary pixel is chosen by angle from the outlet.
* Means and standard deviations of elevation and slope are weighted by pixel area on the WGS84 ellipsoid;
  the hypsometric integral uses the weighted mean. Elevation and slope must cover the network.
* Input grids must be in longitude and latitude (degrees), north up, with matching reference systems and grids.
  Declared no-data values are honoured; a mask that hides pixels holding values is refused.
* Invalid thresholds, non-finite inputs at targets, grids of 2^31 pixels or more and unmeasurable catchments are
  refused with a message. Distances stay finite between opposite points on the Earth.
* Output files carry the exact grid, use BigTIFF when needed and do not overwrite an existing file. Arrays kept
  or returned by `FlowMorph` are read-only; take a `.copy()` to change one.
* Tests cover plan form, raster validation and area thresholds, with reference perimeters for the bundled basin
  and made-up catchments.

The changes before 1.1 are in [`CHANGELOG.md` at `v1.0.0`](https://github.com/hellolulujiang/flowmorph/blob/v1.0.0/CHANGELOG.md).
