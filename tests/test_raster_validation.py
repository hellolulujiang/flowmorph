"""Small regressions for spatial alignment and distances at the edge of their domain."""

import math

import numpy as np
import pytest
import rasterio

from flowmorph import FlowMorph, earth, raster


TRANSFORM = rasterio.Affine(0.01, 0.0, 2.0, 0.0, -0.01, 49.0)


def _write(path, values, crs="EPSG:4326", transform=TRANSFORM):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                       dtype=values.dtype, transform=transform, crs=crs) as dataset:
        dataset.write(values, 1)
    return path


def _inputs(tmp_path, different_crs=None):
    grids = {
        "dir": np.array([[1, 0]], dtype=np.uint8),
        "upa": np.array([[1, 2]], dtype=np.float32),
        "elv": np.array([[2, 1]], dtype=np.float32),
        "seq": np.array([[0, 1]], dtype=np.int32),
        "upg": np.array([[1, 2]], dtype=np.int64),
    }
    grids["ring"] = np.pad(grids["elv"], 1, constant_values=-9999)
    paths = {}
    for name, values in grids.items():
        transform = TRANSFORM * rasterio.Affine.translation(-1, -1) if name == "ring" else TRANSFORM
        # NAD83 is also geographic in degrees: degree validation alone cannot catch this mismatch.
        crs = "EPSG:4269" if name == different_crs else "EPSG:4326"
        paths[name] = _write(tmp_path / f"{name}.tif", values, crs, transform)
    return paths


def _from_paths(paths):
    return FlowMorph.from_raster(paths["dir"], paths["upa"], elv_path=paths["elv"], ring_path=paths["ring"],
                                 seq_dfs_path=paths["seq"], upg_path=paths["upg"], min_upstream_area_km2=0.1)


@pytest.mark.parametrize("different_crs", ["upa", "elv", "ring", "seq", "upg"])
def test_every_companion_raster_must_have_the_direction_crs(tmp_path, different_crs):
    with pytest.raises(ValueError, match="same CRS"):
        _from_paths(_inputs(tmp_path, different_crs))


def test_matching_rasters_keep_their_values_and_index(tmp_path):
    model = _from_paths(_inputs(tmp_path))
    np.testing.assert_array_equal(model.upa, [[1, 2]])
    np.testing.assert_array_equal(model.seq_dfs, [0, 1])
    np.testing.assert_array_equal(model.upg, [1, 2])
    np.testing.assert_array_equal(model.ring[1:-1, 1:-1], model.elevation)
    assert float(model.upstream_terrain()["mean_elv"][0, 1]) == 1.5


def test_geographic_raster_in_grads_is_not_a_degree_grid(tmp_path):
    path = _write(tmp_path / "grad_dir.tif", np.array([[1, 0]], dtype=np.uint8), "EPSG:4807")
    with pytest.raises(ValueError, match="angular units in degrees"):
        raster.read_d8(path)


def test_array_api_also_refuses_grads():
    with pytest.raises(ValueError, match="angular units in degrees"):
        FlowMorph(np.array([[1, 0]], dtype=np.uint8), np.array([[1, 2]], dtype=np.float32),
                  TRANSFORM.to_gdal(), crs="EPSG:4807")


@pytest.mark.parametrize("crs,transform,reason", [
    ("EPSG:4326", rasterio.Affine(.01, .02, 2, 0, -.01, 49), "north up"),
    ("EPSG:4326", rasterio.Affine(.01, 0, 2, .03, -.01, 49), "north up"),
    ("EPSG:4326", rasterio.Affine(.01, 0, 2, 0, .01, 49), "north up"),
    ("EPSG:3857", TRANSFORM, "longitude-latitude"),
    (None, TRANSFORM, "longitude-latitude"),
])
def test_integer_raster_must_be_geographic_and_north_up(tmp_path, crs, transform, reason):
    path = _write(tmp_path / "seq.tif", np.array([[0, 1]], dtype=np.int32), crs, transform)
    with pytest.raises(ValueError, match=reason):
        raster.read_int(path, (1, 2), TRANSFORM.to_gdal())


@pytest.mark.parametrize("model", [earth.SPHERE, earth.WGS84])
def test_antipodal_distance_stays_finite(model):
    distance = earth.distance_m(8.0, 0.0, -8.0, 180.0, model)
    assert math.isfinite(distance)
    assert 19_800_000.0 < distance < 20_200_000.0
    assert distance == pytest.approx(earth.distance_m(-8.0, 180.0, 8.0, 0.0, model))
    assert earth.distance_m(8.0, 0.0, 8.0, 0.0, model) == 0.0
    if model == earth.SPHERE:
        assert distance == pytest.approx(math.pi * earth.R_AUTHALIC)


def _write_with_mask(path, values, hidden, nodata=None):
    """a raster with an internal mask band over the pixels of `hidden`"""
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                           dtype=values.dtype, transform=TRANSFORM, crs="EPSG:4326", nodata=nodata) as dataset:
            dataset.write(values, 1)
            dataset.write_mask(np.where(hidden, 0, 255).astype(np.uint8))
    return path


def test_a_mask_over_cells_that_hold_values_is_refused(tmp_path):
    """the C programs read the values and know no mask: two sinks, one of them hidden by a mask band, would be one
    sink here and two there"""
    sinks = np.array([[0, 0]], dtype=np.uint8)
    hidden = np.array([[False, True]])
    with pytest.raises(ValueError, match="mask hides 1 pixels that hold values"):
        raster.read_d8(_write_with_mask(tmp_path / "dir_masked.tif", sinks, hidden))
    with pytest.raises(ValueError, match="mask hides"):
        raster.read_float(_write_with_mask(tmp_path / "upa_masked.tif", np.array([[20, 20]], dtype=np.float32), hidden))
    with pytest.raises(ValueError, match="mask hides"):
        raster.read_int(_write_with_mask(tmp_path / "seq_masked.tif", np.array([[0, 1]], dtype=np.int32), hidden))


def test_a_mask_over_the_packages_own_no_data_is_let_through(tmp_path):
    """no no-data declared, a mask band over pixels that hold 247 / -9999 / -1: the C code, reading the values, takes
    them for no data as well"""
    hidden = np.array([[False, True]])
    d8, _, _ = raster.read_d8(_write_with_mask(tmp_path / "dir.tif", np.array([[0, 247]], dtype=np.uint8), hidden))
    assert d8.tolist() == [[0, 247]]
    floats = raster.read_float(_write_with_mask(tmp_path / "upa.tif", np.array([[20, -9999]], dtype=np.float32),
                                                hidden))
    assert floats.tolist() == [[20.0, -9999.0]]
    index = raster.read_int(_write_with_mask(tmp_path / "seq.tif", np.array([[0, -1]], dtype=np.int32), hidden))
    assert index.tolist() == [[0, -1]]
    with pytest.raises(ValueError, match="mask hides"):       # another value under the mask is still refused
        raster.read_int(_write_with_mask(tmp_path / "seq_other.tif", np.array([[0, 7]], dtype=np.int32), hidden))


def test_the_declared_no_data_is_no_data_under_a_mask_that_calls_it_valid(tmp_path):
    """a file with a no-data value and a mask band of valid pixels only: the readers judge by the value and take it
    for no data, the elevation ring as well"""
    nowhere = np.array([[False, False]])
    floats = raster.read_float(_write_with_mask(tmp_path / "elv.tif", np.array([[20, -32768]], dtype=np.float32),
                                                nowhere, nodata=-32768))
    assert floats.tolist() == [[20.0, -9999.0]]
    index = raster.read_int(_write_with_mask(tmp_path / "seq.tif", np.array([[0, -5]], dtype=np.int32), nowhere,
                                             nodata=-5))
    assert index.tolist() == [[0, -1]]
    ring = raster.read_ring(_write_with_mask(tmp_path / "ring.tif", np.array([[20, -32768]], dtype=np.float32),
                                             nowhere, nodata=-32768), (1, 2), TRANSFORM.to_gdal())
    assert -32768.0 not in ring.tolist()[1] and ring[1, 1] == 20.0


def test_a_file_that_is_there_is_not_written_over(tmp_path):
    path = str(tmp_path / "out.tif")
    raster.write_float(path, np.array([[1.0, 2.0]], dtype=np.float32), TRANSFORM.to_gdal(), "EPSG:4326")
    with pytest.raises(FileExistsError):
        raster.write_float(path, np.array([[3.0, 4.0]], dtype=np.float32), TRANSFORM.to_gdal(), "EPSG:4326")
    assert raster.read_float(path).tolist() == [[1.0, 2.0]]


def test_a_mask_that_is_the_no_data_value_reads_as_before(tmp_path):
    """MERIT Hydro's way: the no data is a value of the raster (247 for the directions), and its mask says no more"""
    directions = np.array([[1, 247]], dtype=np.uint8)
    path = tmp_path / "dir_nodata.tif"
    with rasterio.open(path, "w", driver="GTiff", height=1, width=2, count=1, dtype="uint8", transform=TRANSFORM,
                       crs="EPSG:4326", nodata=247) as dataset:
        dataset.write(directions, 1)
    d8, _, _ = raster.read_d8(path)
    assert d8.tolist() == [[1, 247]]
    floats = np.array([[3.5, -9999.0]], dtype=np.float32)
    path = tmp_path / "upa_nodata.tif"
    with rasterio.open(path, "w", driver="GTiff", height=1, width=2, count=1, dtype="float32", transform=TRANSFORM,
                       crs="EPSG:4326", nodata=-9999.0) as dataset:
        dataset.write(floats, 1)
    assert raster.read_float(path).tolist() == [[3.5, -9999.0]]
