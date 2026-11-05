"""
Satellite Phased-Array Performance Analysis

Calculates:
- Peak beam direction
- Pointing error
- Half-power beamwidth (HPBW)
- Peak sidelobe level (SLL)

Measurements are calculated from the phi = 0 degree.
"""

import numpy as np
from scipy.signal import find_peaks

import antenna_config as config
from radiation_pattern import calculate_total_pattern


def interpolate_crossing(x1, y1, x2, y2, target):
    #Linearly interpolate the angle at a target power level

    if y2 == y1:
        return (x1 + x2) / 2

    return x1 + (target - y1) * (x2 - x1) / (y2 - y1)


def interpolate_peak(x_values, y_values, peak_index):
    """
    Estimate the peak position using parabolic interpolation using the three samples surrounding the detected peak.
    Returns:
        refined_peak_x: Estimated peak angle
        refined_peak_y: Estimated peak value
    """

    # Interpolation requires one sample on either side
    if peak_index <= 0 or peak_index >= len(y_values) - 1:
        return x_values[peak_index], y_values[peak_index]

    y_left = y_values[peak_index - 1]
    y_center = y_values[peak_index]
    y_right = y_values[peak_index + 1]

    denominator = y_left - 2 * y_center + y_right

    if abs(denominator) < 1e-12:
        return x_values[peak_index], y_values[peak_index]

    # Fractional position of the estimated peak
    offset = 0.5 * (y_left - y_right) / denominator

    # Keep interpolation within the neighbouring samples
    offset = np.clip(offset, -1.0, 1.0)

    step = x_values[peak_index + 1] - x_values[peak_index]

    refined_peak_x = x_values[peak_index] + offset * step

    # Estimate peak amplitude from the fitted parabola
    refined_peak_y = (
        y_center
        - 0.25 * (y_left - y_right) * offset
    )

    return refined_peak_x, refined_peak_y


def calculate_performance():
    #Calculate antenna performance metrics

    # Generate a higher-resolution angular sweep
    theta_values = np.linspace(0, 90, 721)

    phi_values = np.array([0.0])

    theta_grid, phi_grid, power_pattern = calculate_total_pattern(
        theta_values,
        phi_values
    )

    theta = theta_grid[:, 0]
    power = power_pattern[:, 0]

    # Convert normalized power to dB
    power_db = 10 * np.log10(np.maximum(power, 1e-12))

    # Locate the main beam peak
    peak_index = np.argmax(power)

    peak_theta, peak_power_db = interpolate_peak(
        theta,
        power_db,
        peak_index
    )

    # Pointing error relative to commanded direction
    pointing_error = abs(
        peak_theta - config.STEERING_THETA_DEG
    )

    # Half-power beamwidth (HPBW) calculation

    half_power = 0.5

    # Search left of the main beam for the -3 dB crossing
    left_index = peak_index

    while left_index > 0 and power[left_index] >= half_power:
        left_index -= 1

    # Search right of the main beam
    right_index = peak_index

    while (
        right_index < len(power) - 1
        and power[right_index] >= half_power
    ):
        right_index += 1

    if left_index == 0 or right_index == len(power) - 1:
        raise ValueError(
            "Could not locate both half-power beamwidth crossings."
        )

    left_crossing = interpolate_crossing(
        theta[left_index],
        power[left_index],
        theta[left_index + 1],
        power[left_index + 1],
        half_power
    )

    right_crossing = interpolate_crossing(
        theta[right_index - 1],
        power[right_index - 1],
        theta[right_index],
        power[right_index],
        half_power
    )

    hpbw = right_crossing - left_crossing

    # Peak sidelobe level (SLL) calculation

    # Find the first local minima on either side of the main beam.
    # These approximate the first nulls that bound the main lobe.

    minima, _ = find_peaks(-power)

    left_minima = minima[minima < peak_index]
    right_minima = minima[minima > peak_index]

    if len(left_minima) == 0 or len(right_minima) == 0:
        raise ValueError(
            "Could not locate main-lobe boundaries for sidelobe analysis."
        )

    left_null = left_minima[-1]
    right_null = right_minima[0]

    # Find local maxima outside the main lobe
    sidelobe_indices, _ = find_peaks(power)

    sidelobe_indices = sidelobe_indices[
        (sidelobe_indices < left_null)
        | (sidelobe_indices > right_null)
    ]

    if len(sidelobe_indices) == 0:
        peak_sll_db = None
    else:
        peak_sll_db = np.max(
            power_db[sidelobe_indices]
        ) - peak_power_db

    # Return the calculated performance metrics

    return {
        "peak_theta_deg": peak_theta,
        "commanded_theta_deg": config.STEERING_THETA_DEG,
        "pointing_error_deg": pointing_error,
        "hpbw_deg": hpbw,
        "peak_sll_db": peak_sll_db,
        "left_half_power_deg": left_crossing,
        "right_half_power_deg": right_crossing,
        "left_null_deg": theta[left_null],
        "right_null_deg": theta[right_null]
    }


def display_performance(results):
    """Print the calculated antenna performance."""

    print("=" * 60)
    print("Satellite Phased-Array Antenna Performance Analysis")
    print("=" * 60)

    print(f"Array configuration: {config.ELEMENTS_X} x {config.ELEMENTS_Y}")
    print(f"Operating frequency: {config.FREQUENCY_HZ / 1e9:.2f} GHz")

    print("-" * 60)

    print(
        f"Commanded beam direction: "
        f"{results['commanded_theta_deg']:.3f} degrees"
    )

    print(
        f"Measured beam direction: "
        f"{results['peak_theta_deg']:.3f} degrees"
    )

    print(
        f"Pointing error: "
        f"{results['pointing_error_deg']:.3f} degrees"
    )

    print(
        f"Half-power beamwidth: "
        f"{results['hpbw_deg']:.3f} degrees"
    )

    if results["peak_sll_db"] is not None:
        print(
            f"Peak sidelobe level: "
            f"{results['peak_sll_db']:.2f} dB"
        )
    else:
        print("Peak sidelobe level: Not detected")

    print("-" * 60)

    print(
        f"Left -3 dB crossing: "
        f"{results['left_half_power_deg']:.3f} degrees"
    )

    print(
        f"Right -3 dB crossing: "
        f"{results['right_half_power_deg']:.3f} degrees"
    )

    print(
        f"Approximate first nulls: "
        f"{results['left_null_deg']:.3f}° / "
        f"{results['right_null_deg']:.3f}°"
    )

    print("=" * 60)


if __name__ == "__main__":
    results = calculate_performance()
    display_performance(results)
