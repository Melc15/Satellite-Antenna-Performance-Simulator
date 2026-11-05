"""
Satellite Phased-Array Frequency Sweep

Evaluates antenna performance across multiple operating frequencies.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import antenna_config as config

from performance_analysis import (
    interpolate_crossing,
    interpolate_peak
)
from radiation_pattern import calculate_total_pattern

# Frequencies to evaluate (GHz)

FREQUENCIES_GHZ = np.array([
    18.0,
    19.0,
    20.0,
    21.0,
    22.0
])

THETA_RESOLUTION = 1801


#Performance metrics calculation for a single frequency

def calculate_frequency_performance(frequency_ghz):

    frequency_hz = frequency_ghz * 1e9

    theta_values = np.linspace(
        0,
        90,
        THETA_RESOLUTION
    )

    phi_values = np.array([0.0])

    theta_grid, phi_grid, power_pattern = calculate_total_pattern(
        theta_values,
        phi_values,
        frequency_hz=frequency_hz
    )

    theta = theta_grid[:, 0]
    power = power_pattern[:, 0]

    power_db = 10 * np.log10(
        np.maximum(power, 1e-12)
    )

    # Locate main beam
    peak_index = np.argmax(power)

    peak_theta, peak_power_db = interpolate_peak(
        theta,
        power_db,
        peak_index
    )

    pointing_error = abs(
        peak_theta - config.STEERING_THETA_DEG
    )

    # Half-Power Beamwidth (HPBW) calculation

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

    # Peak Sidelobe Level (SLL) calculation

    from scipy.signal import find_peaks

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
        "frequency_ghz": frequency_ghz,
        "peak_theta_deg": peak_theta,
        "pointing_error_deg": pointing_error,
        "hpbw_deg": hpbw,
        "peak_sll_db": peak_sll_db
    }

# Run frequency sweep across all specified frequencies

def run_frequency_sweep():

    print("=" * 70)
    print("Satellite Phased-Array Frequency Sweep")
    print("=" * 70)

    print(
        f"Array configuration: "
        f"{config.ELEMENTS_X} x {config.ELEMENTS_Y}"
    )

    print(
        f"Total elements: {config.TOTAL_ELEMENTS}"
    )

    print(
        f"Reference frequency: "
        f"{config.FREQUENCY_HZ / 1e9:.2f} GHz"
    )

    print(
        f"Commanded steering angle: "
        f"{config.STEERING_THETA_DEG:.1f} degrees"
    )

    print("Physical element spacing: FIXED")
    print("-" * 70)

    results = []

    for frequency in FREQUENCIES_GHZ:

        print(
            f"Calculating performance at "
            f"{frequency:.1f} GHz..."
        )

        result = calculate_frequency_performance(
            frequency
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    # Save numerical results
    results_df.to_csv(
        "frequency_sweep_results.csv",
        index=False
    )

    print("\nFrequency Sweep Results:")
    print("-" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "frequency_ghz": "{:.1f}".format,
                "peak_theta_deg": "{:.3f}".format,
                "pointing_error_deg": "{:.3f}".format,
                "hpbw_deg": lambda x: (
                    f"{x:.3f}" if pd.notna(x) else "N/A"
                ),
                "peak_sll_db": lambda x: (
                    f"{x:.2f}" if pd.notna(x) else "N/A"
                )
            }
        )
    )

    print("-" * 70)

    generate_frequency_plots(results_df)

    print("\nFrequency sweep completed. Performance plots saved as PNG files.")
    print("=" * 70)

    print("Saved:")
    print("- frequency_sweep_results.csv")
    print("- frequency_pointing_error.png")
    print("- frequency_beamwidth.png")
    print("- frequency_sidelobe_level.png")

    return results_df


# Generate plots for frequency sweep results

def generate_frequency_plots(results_df):

    frequencies = results_df["frequency_ghz"]

    # Pointing error
    plt.figure(figsize=(9, 5))

    plt.plot(
        frequencies,
        results_df["pointing_error_deg"],
        marker="o",
        linewidth=2
    )

    plt.axvline(
        config.FREQUENCY_HZ / 1e9,
        linestyle="--",
        label="20 GHz reference"
    )

    plt.title("Pointing Error vs Operating Frequency")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Pointing Error (degrees)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "frequency_pointing_error.png",
        dpi=300
    )

    plt.close()

    # Half-power beamwidth
    plt.figure(figsize=(9, 5))

    plt.plot(
        frequencies,
        results_df["hpbw_deg"],
        marker="o",
        linewidth=2
    )

    plt.axvline(
        config.FREQUENCY_HZ / 1e9,
        linestyle="--",
        label="20 GHz reference"
    )

    plt.title("Half-Power Beamwidth vs Operating Frequency")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("HPBW (degrees)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "frequency_beamwidth.png",
        dpi=300
    )

    plt.close()

    # Peak sidelobe level
    plt.figure(figsize=(9, 5))

    plt.plot(
        frequencies,
        results_df["peak_sll_db"],
        marker="o",
        linewidth=2
    )

    plt.axvline(
        config.FREQUENCY_HZ / 1e9,
        linestyle="--",
        label="20 GHz reference"
    )

    plt.title("Peak Sidelobe Level vs Operating Frequency")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Peak Sidelobe Level (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "frequency_sidelobe_level.png",
        dpi=300
    )

    plt.close()


#MAIN

if __name__ == "__main__":

    run_frequency_sweep()
