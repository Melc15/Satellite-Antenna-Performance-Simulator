
"""
Satellite Phased-Array Beam-Steering Sweep

Evaluates antenna performance at multiple commanded beam-steering angles at a fixed operating frequency.

Uses parabolic interpolation to refine the estimated main-beam peak between angular grid samples.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

import antenna_config as config
from radiation_pattern import calculate_total_pattern
from performance_analysis import (
    interpolate_crossing,
    interpolate_peak
)

# Sweep configuration

STEERING_ANGLES_DEG = [0, 10, 20, 30, 40, 50]

OPERATING_FREQUENCY_HZ = config.FREQUENCY_HZ

# 0.1-degree observation grid
THETA_VALUES = np.linspace(-90, 90, 1801)

PHI_VALUES = np.array([0.0])

# Calculate performance metrics for a single commanded steering angle

def calculate_steering_performance(commanded_angle):

    theta_grid, phi_grid, power_pattern = calculate_total_pattern(
        THETA_VALUES,
        PHI_VALUES,
        frequency_hz=OPERATING_FREQUENCY_HZ,
        steering_theta_deg=commanded_angle,
        steering_phi_deg=config.STEERING_PHI_DEG
    )

    theta = theta_grid[:, 0]
    power = power_pattern[:, 0]

    power_db = 10 * np.log10(np.maximum(power, 1e-12))

    # Main beam peak with parabolic interpolation
    peak_index = np.argmax(power)

    peak_theta, peak_power_db = interpolate_peak(
        theta,
        power_db,
        peak_index
    )

    pointing_error = abs(peak_theta - commanded_angle)

    # Half-power beamwidth
    half_power = 0.5

    left_index = peak_index

    while left_index > 0 and power[left_index] >= half_power:
        left_index -= 1

    right_index = peak_index

    while (
        right_index < len(power) - 1
        and power[right_index] >= half_power
    ):
        right_index += 1

    if left_index == 0 or right_index == len(power) - 1:
        hpbw = None

    else:

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

    # Peak sidelobe level
    minima, _ = find_peaks(-power)

    left_minima = minima[minima < peak_index]
    right_minima = minima[minima > peak_index]

    if len(left_minima) == 0 or len(right_minima) == 0:

        peak_sll_db = None

    else:

        left_null = left_minima[-1]
        right_null = right_minima[0]

        sidelobe_indices, _ = find_peaks(power)

        sidelobe_indices = sidelobe_indices[
            (sidelobe_indices < left_null)
            | (sidelobe_indices > right_null)
        ]

        if len(sidelobe_indices) == 0:
            peak_sll_db = None

        else:
            peak_sll_db = (
                np.max(power_db[sidelobe_indices])
                - peak_power_db
            )

    return {
        "commanded_angle_deg": commanded_angle,
        "measured_peak_deg": peak_theta,
        "pointing_error_deg": pointing_error,
        "hpbw_deg": hpbw,
        "peak_sll_db": peak_sll_db
    }


#Generate plots for beam-steering performance metrics

def generate_steering_plots(results_df):

    angles = results_df["commanded_angle_deg"]

    # Pointing error
    plt.figure(figsize=(9, 5))
    plt.plot(
        angles,
        results_df["pointing_error_deg"],
        marker="o",
        linewidth=2
    )
    plt.title("Beam Pointing Error vs Commanded Steering Angle")
    plt.xlabel("Commanded Steering Angle (degrees)")
    plt.ylabel("Pointing Error (degrees)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("beam_steering_pointing_error.png", dpi=300)
    plt.close()

    # Half-power beamwidth
    plt.figure(figsize=(9, 5))
    plt.plot(
        angles,
        results_df["hpbw_deg"],
        marker="o",
        linewidth=2
    )
    plt.title("Half-Power Beamwidth vs Steering Angle")
    plt.xlabel("Commanded Steering Angle (degrees)")
    plt.ylabel("HPBW (degrees)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("beam_steering_beamwidth.png", dpi=300)
    plt.close()

    # Peak sidelobe level
    plt.figure(figsize=(9, 5))
    plt.plot(
        angles,
        results_df["peak_sll_db"],
        marker="o",
        linewidth=2
    )
    plt.title("Peak Sidelobe Level vs Steering Angle")
    plt.xlabel("Commanded Steering Angle (degrees)")
    plt.ylabel("Peak Sidelobe Level (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("beam_steering_sidelobe_level.png", dpi=300)
    plt.close()


#Run the beam-steering sweep and save results

def run_beam_steering_sweep():

    print("=" * 72)
    print("SATELLITE PHASED-ARRAY BEAM-STEERING SWEEP")
    print("=" * 72)

    print(f"Array configuration: {config.ELEMENTS_X} x {config.ELEMENTS_Y}")
    print(f"Total elements: {config.TOTAL_ELEMENTS}")
    print(f"Operating frequency: {OPERATING_FREQUENCY_HZ / 1e9:.2f} GHz")
    print(f"Azimuth angle: {config.STEERING_PHI_DEG:.1f} degrees")
    print("Frequency and physical element spacing: FIXED")
    print("Peak estimation: PARABOLIC INTERPOLATION")
    print("-" * 72)

    results = []

    for angle in STEERING_ANGLES_DEG:

        print(f"Calculating performance at {angle:.1f} degrees...")

        results.append(calculate_steering_performance(angle))

    results_df = pd.DataFrame(results)

    results_df.to_csv("beam_steering_results.csv", index=False)

    print("\nBeam-steering sweep completed. Results saved to 'beam_steering_results.csv'.\n")
    print("-" * 72)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "commanded_angle_deg": "{:.1f}".format,
                "measured_peak_deg": "{:.4f}".format,
                "pointing_error_deg": "{:.4f}".format,
                "hpbw_deg": lambda x: f"{x:.3f}" if pd.notna(x) else "N/A",
                "peak_sll_db": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            }
        )
    )

    print("-" * 72)

    generate_steering_plots(results_df)

    print("\nBeam-steering performance plots saved as PNG files.")
    print("=" * 72)

    print("Saved:")
    print("- beam_steering_results.csv")
    print("- beam_steering_pointing_error.png")
    print("- beam_steering_beamwidth.png")
    print("- beam_steering_sidelobe_level.png")

    return results_df


if __name__ == "__main__":
    run_beam_steering_sweep()
