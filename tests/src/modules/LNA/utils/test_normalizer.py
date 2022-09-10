import numpy as np

from src.modules.LNA.utils.Normalization import clamp, z_inverse_normalize, z_normalize


def test_it_should_be_able_to_convert_to_a_range():

    value = 0
    value_normalized = z_normalize(a=-1, b=1, x=value, x_max=1, x_min=0)
    assert np.isclose(-1, value_normalized)
    value = 1
    value_normalized = z_normalize(a=-1, b=1, x=value, x_max=1, x_min=0)
    assert np.isclose(1, value_normalized)


def test_it_should_be_able_to_invert_the_normalization():
    value = -1
    value_denormalized = z_inverse_normalize(a=-1, b=1, x=value, x_max=1, x_min=0)
    assert np.isclose(value_denormalized, 0)


def test_it_should_be_able_to_clamp_the_value():
    max_value = 1
    min_value = -1
    value = 2
    value_clamped = clamp(value=value, max_=max_value, min_=min_value)

    assert np.isclose(value_clamped, max_value)
    value = -2
    value_clamped = clamp(value=value, max_=max_value, min_=min_value)
    assert np.isclose(value_clamped, min_value)
