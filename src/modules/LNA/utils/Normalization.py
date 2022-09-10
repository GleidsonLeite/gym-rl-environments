def z_normalize(
    x: float,
    x_min: float,
    x_max: float,
    a: float,
    b: float,
) -> float:
    return a + (x - x_min) * (b - a) / (x_max - x_min)


def z_inverse_normalize(
    x: float,
    x_min: float,
    x_max: float,
    a: float,
    b: float,
) -> float:
    return (x - a) * (x_max - x_min) / (b - a) + x_min


def clamp(value: float, max_: float, min_: float) -> float:
    return max(min(max_, value), min_)
