import numpy as np
import pytest

from flowmorph import FlowMorph

TRANSFORM = (0., .01, 0., 10., 0., -.01)
D8 = np.array([[1, 0]], dtype=np.uint8)
UPA = np.array([[1., 2.]], dtype=np.float32)


@pytest.mark.parametrize("threshold", [1e-50, 1e100, -1., 0., np.nan, np.inf])
def test_unrepresentable_target_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="min_upstream_area_km2"):
        FlowMorph(D8, UPA, TRANSFORM, min_upstream_area_km2=threshold)


@pytest.mark.parametrize("threshold,expected", [
    (1., [True, True]), (2., [False, True]),
    (np.finfo(np.float32).smallest_subnormal, [True, True]),
    (np.finfo(np.float32).max, [False, False]),
])
def test_representable_threshold_keeps_float32_target_selection(threshold, expected):
    model = FlowMorph(D8, UPA, TRANSFORM, min_upstream_area_km2=threshold)
    np.testing.assert_array_equal(model.is_target, expected)
