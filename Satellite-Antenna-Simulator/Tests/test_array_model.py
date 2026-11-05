
import numpy as np

from array_model import (
    get_element_positions,
    calculate_steering_phases,
    calculate_array_factor
)

import antenna_config as config


def test_element_count():
    x, y = get_element_positions()

    assert len(x) == 256
    assert len(y) == 256


def test_element_positions_are_centred():
    x, y = get_element_positions()

    assert np.isclose(np.mean(x), 0.0)
    assert np.isclose(np.mean(y), 0.0)


def test_steering_phases_have_correct_length():
    phases = calculate_steering_phases()

    assert len(phases) == config.TOTAL_ELEMENTS


def test_array_factor_is_normalized():
    theta = np.linspace(0, 90, 181)
    phi = np.linspace(-180, 180, 361)

    _, _, pattern = calculate_array_factor(theta, phi)

    assert np.isclose(np.max(pattern), 1.0)
    assert np.all(pattern >= 0)
    assert np.all(pattern <= 1.0)


def test_beam_steering_direction():
    theta = np.linspace(0, 90, 181)
    phi = np.linspace(-180, 180, 361)

    theta_grid, phi_grid, pattern = calculate_array_factor(
        theta,
        phi
    )

    peak = np.unravel_index(
        np.argmax(pattern),
        pattern.shape
    )

    assert abs(
        theta_grid[peak] - config.STEERING_THETA_DEG
    ) <= 0.5

    assert abs(phi_grid[peak]) <= 0.5
