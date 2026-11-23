"""Small statistics shared by the calibration and audit code."""

import math


def wilson(p: float | None, n: int, z: float = 1.959964) -> list[float] | None:
    if p is None or n == 0:
        return None
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]
