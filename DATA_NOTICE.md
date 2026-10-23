# Notice on the bundled data

The MIT terms in [`LICENSE`](LICENSE) cover the code and the documentation in this repository:
`flowmorph/`, `tests/` (but not `tests/reference/expected_c.npz` and `tests/reference/expected_c_perimeter.npz`),
and the files at the top level. They do not cover the files in `data/` and those two files.

`data/dir_example.tif`, `data/upa_example.tif` and `data/elv_example.tif` are an excerpt of MERIT Hydro, 376 rows by
434 columns at 3 arc-seconds around one basin in Tasmania (Yamazaki et al., 2019,
<https://doi.org/10.1029/2019WR024873>): the flow directions and the upstream area of the basin's pixels, and the
elevation of the whole window. Its authors distribute MERIT Hydro under CC BY-NC 4.0 or ODbL 1.0
(<https://global-hydrodynamics.github.io/MERIT_Hydro/>); the excerpt is redistributed here under CC BY-NC 4.0, for
reproducibility, with attribution to its authors. `tests/reference/expected_c.npz` and
`tests/reference/expected_c_perimeter.npz` hold values the C code of FlowMorph computed from MERIT Hydro over the same
window, and carry the same terms. `tests/reference/expected_c_vs_cells_synthetic.npz` holds the perimeters the C code
computed for made-up catchments, and is not drawn from MERIT Hydro.
