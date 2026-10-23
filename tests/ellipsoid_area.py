"""The area of a pixel of every row of a longitude-latitude grid on the WGS84 ellipsoid, for the tests: integrated
here, so that an expected value does not come from the function it checks."""

import numpy as np


def area_of_rows(north, y_resolution, x_resolution, nrow):
    """The area in m^2 of a pixel of every row, integrated over the row's latitudes with Simpson's rule on 64
    intervals: a^2 (1 - e^2) cos(phi) / (1 - e^2 sin^2(phi))^2 dphi dlambda."""
    semi_major = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity2 = flattening * (2.0 - flattening)
    simpson = np.ones(65)
    simpson[1:-1:2] = 4.0
    simpson[2:-1:2] = 2.0
    areas = np.empty(nrow)
    for row in range(nrow):
        latitudes = np.radians(np.linspace(north + (row + 1) * y_resolution, north + row * y_resolution, 65))
        values = np.cos(latitudes) / (1.0 - eccentricity2 * np.sin(latitudes) ** 2) ** 2
        areas[row] = (semi_major ** 2 * (1.0 - eccentricity2) * np.radians(abs(x_resolution))
                      * (latitudes[1] - latitudes[0]) / 3.0 * float((simpson * values).sum()))
    return areas
