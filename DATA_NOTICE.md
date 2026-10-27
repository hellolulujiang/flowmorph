# Notice on the bundled data

The MIT terms in [`LICENSE`](LICENSE) cover the code and the documentation in this repository:
`flowmorph/`, `tests/` (but not `tests/reference/expected.npz`), and the files at the top level. They do not cover
the files in `data/`, that file, and the three figures in `docs/media/`.

`data/dir_example.tif`, `data/upa_example.tif` and `data/elv_example.tif` are an excerpt of 294 rows by 616 columns at
3 arc-seconds around the Nanxi River basin (Fujian, China), the example basin of the FullBasin paper: the flow
directions and the upstream area of MERIT Hydro v1.0.1 (Yamazaki et al., 2019, <https://doi.org/10.1029/2019WR024873>)
on the basin's pixels, and the elevation of MERIT DEM v1.0.3 (Yamazaki et al., 2017,
<https://doi.org/10.1002/2017GL072874>) on the whole window. Their authors distribute MERIT Hydro and MERIT DEM under
CC BY-NC 4.0 or ODbL 1.0, at the user's choice, terms that also hold for data derived from them
(<https://global-hydrodynamics.github.io/MERIT_Hydro/>, <https://global-hydrodynamics.github.io/MERIT_DEM/>); the
excerpt is redistributed here under the same two licences, either of them at your choice, for reproducibility, with
attribution to its authors. `tests/reference/expected.npz` holds the sixteen attributes of FullMorph v1.1 on this
basin, derived from MERIT Hydro and MERIT DEM, and carries the same terms.
`tests/reference/expected_perimeter_synthetic.npz` holds reference perimeters of made-up catchments, and is not drawn
from MERIT Hydro.

The three figures in `docs/media/` are Figures 7, 8 and 9 of the FullBasin paper as revised for Earth System Science
Data (discussion paper: Jiang et al., <https://doi.org/10.5194/essd-2026-402>), drawn on the same basin from MERIT
Hydro and MERIT DEM, and carry the terms of those data.
