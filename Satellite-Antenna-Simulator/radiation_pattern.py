"""
Satellite Phased-Array Radiation Pattern Visualization

Generates:
1. 2D polar radiation pattern
2. 3D radiation pattern

The model combines the ideal array factor with an
assumed cosine-power individual-element pattern.

These are normalized patterns, not measured gain in dBi.
"""

import numpy as np
import matplotlib.pyplot as plt

import antenna_config as config
from array_model import calculate_array_factor



def calculate_total_pattern(
    theta_values,
    phi_values,
    frequency_hz=None,
    steering_theta_deg=None,
    steering_phi_deg=None
):
    """
    Combine the array factor with the individual-element pattern to produce the total radiation pattern. 
    Supports frequency-dependent and beam-steering calculations and returns normalized power between 0 and 1.
    """

    theta_grid, phi_grid, array_factor = calculate_array_factor(
        theta_values,
        phi_values,
        steering_theta_deg=steering_theta_deg,
        steering_phi_deg=steering_phi_deg,
        frequency_hz=frequency_hz
    )

    theta_radians = np.radians(theta_grid)

    # Simplified individual-element power pattern
    element_pattern = np.maximum(
        np.cos(theta_radians),
        0.0
    ) ** config.ELEMENT_PATTERN_EXPONENT

    # Convert normalized field magnitude to relative power
    total_power = (array_factor ** 2) * element_pattern

    # Normalize combined pattern
    maximum = np.max(total_power)

    if maximum > 0:
        total_power /= maximum

    return theta_grid, phi_grid, total_power



def plot_2d_polar(theta_grid, phi_grid, power_pattern):
    #Plot the radiation pattern in the phi = 0 degree cut.

    # Find the phi column closest to zero
    phi_index = np.argmin(np.abs(phi_grid[0, :]))

    theta_degrees = theta_grid[:, phi_index]
    power = power_pattern[:, phi_index]

    # Convert normalized power to dB
    power_db = 10 * np.log10(np.maximum(power, 1e-12))

    # Limit display range
    power_db = np.maximum(power_db, config.DB_FLOOR)

    # Convert dB values into positive polar radii
    polar_radius = power_db - config.DB_FLOOR

    theta_radians = np.radians(theta_degrees)

    fig, ax = plt.subplots(
        figsize=(9, 9),
        subplot_kw={"projection": "polar"}
    )

    ax.plot(
        theta_radians,
        polar_radius,
        linewidth=2,
        label="Radiation pattern"
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_ylim(0, abs(config.DB_FLOOR))

    tick_values = np.arange(
        config.DB_FLOOR,
        1,
        10
    )

    ax.set_yticks(tick_values - config.DB_FLOOR)
    ax.set_yticklabels([f"{value} dB" for value in tick_values])

    ax.set_title(
        "2D Polar Radiation Pattern\n"
        f"Steering: {config.STEERING_THETA_DEG}° theta, "
        f"{config.STEERING_PHI_DEG}° phi\n"
        "Phi = 0° cut",
        pad=25
    )

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    plt.tight_layout()
    plt.savefig("radiation_pattern_2d.png", dpi=300)
    plt.show()


def plot_3d_pattern(theta_grid, phi_grid, power_pattern):
    #Generate a 3D upper-hemisphere radiation pattern.

    #Radius represents normalized power.

    theta = np.radians(theta_grid)
    phi = np.radians(phi_grid)

    # Radius is normalized power
    radius = power_pattern

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    fig = plt.figure(figsize=(11, 9))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    surface = ax.plot_surface(
        x,
        y,
        z,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.65,
        label="Normalized power"
    )

    ax.set_title(
        "3D Satellite Phased-Array Radiation Pattern\n"
        f"Steering angle: {config.STEERING_THETA_DEG}°"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    plt.savefig("radiation_pattern_3d.png", dpi=300)
    plt.show()


def main():
    print("=" * 55)
    print("Radiation Pattern Visualization")
    print("=" * 55)

    theta_values = np.linspace(
        0,
        90,
        config.THETA_RESOLUTION
    )

    phi_values = np.linspace(
        -180,
        180,
        config.PHI_RESOLUTION
    )

    theta_grid, phi_grid, power_pattern = calculate_total_pattern(
        theta_values,
        phi_values
    )

    peak_index = np.unravel_index(
        np.argmax(power_pattern),
        power_pattern.shape
    )

    peak_theta = theta_grid[peak_index]
    peak_phi = phi_grid[peak_index]

    print(f"Total elements: {config.TOTAL_ELEMENTS}")
    print(f"Operating frequency: {config.FREQUENCY_HZ / 1e9:.2f} GHz")

    print(
        f"Commanded steering angle: "
        f"{config.STEERING_THETA_DEG:.1f}°"
    )

    print(
        f"Combined pattern peak: "
        f"Theta={peak_theta:.1f}°, Phi={peak_phi:.1f}°"
    )

    print("Generating 2D and 3D radiation patterns...")

    plot_2d_polar(
        theta_grid,
        phi_grid,
        power_pattern
    )

    plot_3d_pattern(
        theta_grid,
        phi_grid,
        power_pattern
    )

    print("Saved: radiation_pattern_2d.png")
    print("Saved: radiation_pattern_3d.png")

    print("=" * 55)


if __name__ == "__main__":
    main()
