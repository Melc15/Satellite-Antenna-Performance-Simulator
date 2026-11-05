
"""
Phased-Array Electromagnetic Model

Calculates the ideal array factor of a uniformly excited rectangular planar antenna array.

Supports frequency-dependent calculations while preserving the original 20 GHz default configuration.

"""

import numpy as np

import antenna_config as config


def get_element_positions():
    #Generate centred X and Y element coordinates in metres.

    x_axis = (
        np.arange(config.ELEMENTS_X)
        - (config.ELEMENTS_X - 1) / 2
    ) * config.SPACING_X_METRES

    y_axis = (
        np.arange(config.ELEMENTS_Y)
        - (config.ELEMENTS_Y - 1) / 2
    ) * config.SPACING_Y_METRES

    x_positions, y_positions = np.meshgrid(
        x_axis,
        y_axis
    )

    return x_positions.flatten(), y_positions.flatten()


def calculate_steering_phases(
    theta_deg=None,
    phi_deg=None,
    frequency_hz=None
):

    #Calculate the phase excitation required to steer the main beam toward a specified direction. The frequency suuplied supports the frequency sweeps.


    if theta_deg is None:
        theta_deg = config.STEERING_THETA_DEG

    if phi_deg is None:
        phi_deg = config.STEERING_PHI_DEG

    if frequency_hz is None:
        wave_number = config.WAVE_NUMBER
    else:
        wavelength = config.SPEED_OF_LIGHT / frequency_hz
        wave_number = 2 * np.pi / wavelength

    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    x_positions, y_positions = get_element_positions()

    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)

    phases = -wave_number * (
        x_positions * u
        + y_positions * v
    )

    return phases


def calculate_array_factor(
    theta_values,
    phi_values,
    steering_theta_deg=None,
    steering_phi_deg=None,
    frequency_hz=None
):
    """
    Calculate the normalized array factor over an observation-angle grid. It returns the observation angles and the normalized field magnitude.
        theta_grid: Observation theta angles
        phi_grid: Observation phi angles
        pattern: Normalized field magnitude (0 to 1)
    """

    theta_values = np.asarray(theta_values, dtype=float)
    phi_values = np.asarray(phi_values, dtype=float)

    theta_grid, phi_grid = np.meshgrid(
        theta_values,
        phi_values,
        indexing="ij"
    )

    theta = np.radians(theta_grid).flatten()
    phi = np.radians(phi_grid).flatten()

    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)

    x_positions, y_positions = get_element_positions()

    if frequency_hz is None:
        wave_number = config.WAVE_NUMBER
    else:
        wavelength = config.SPEED_OF_LIGHT / frequency_hz
        wave_number = 2 * np.pi / wavelength

    observation_phase = wave_number * (
        x_positions[:, None] * u[None, :]
        + y_positions[:, None] * v[None, :]
    )

    excitation_phase = calculate_steering_phases(
        steering_theta_deg,
        steering_phi_deg,
        frequency_hz
    )

    total_field = np.sum(
        np.exp(
            1j * (
                observation_phase
                + excitation_phase[:, None]
            )
        ),
        axis=0
    )

    magnitude = np.abs(total_field)

    maximum = np.max(magnitude)

    if maximum > 0:
        magnitude = magnitude / maximum

    pattern = magnitude.reshape(theta_grid.shape)

    return theta_grid, phi_grid, pattern


def display_model_results():
    #Run a basic beam-steering calculation.

    theta_values = np.linspace(0, 90, 91)
    phi_values = np.linspace(-180, 180, 361)

    theta_grid, phi_grid, pattern = calculate_array_factor(
        theta_values,
        phi_values
    )

    peak_index = np.unravel_index(
        np.argmax(pattern),
        pattern.shape
    )

    peak_theta = theta_grid[peak_index]
    peak_phi = phi_grid[peak_index]

    print("=" * 55)
    print("Phased-Array Model Results")
    print("=" * 55)

    print(f"Number of elements: {config.TOTAL_ELEMENTS}")

    print(
        f"Requested steering direction: "
        f"Theta={config.STEERING_THETA_DEG:.1f}°, "
        f"Phi={config.STEERING_PHI_DEG:.1f}°"
    )

    print(
        f"Calculated peak direction: "
        f"Theta={peak_theta:.1f}°, "
        f"Phi={peak_phi:.1f}°"
    )

    print(f"Peak normalized field: {np.max(pattern):.4f}")

    print("=" * 55)


if __name__ == "__main__":
    display_model_results()
