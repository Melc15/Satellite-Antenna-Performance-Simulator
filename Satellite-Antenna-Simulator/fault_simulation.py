
"""
Satellite Phased-Array Fault Injection Simulator

Simulates:
1. Ideal antenna operation
2. Failed antenna elements
3. Random element phase errors
4. Combined element failures and phase errors

"""

import numpy as np
import matplotlib.pyplot as plt

import antenna_config as config
from array_model import (
    get_element_positions,
    calculate_steering_phases
)


#Fault injection parameters

FAILED_ELEMENT_PERCENT = 10.0

PHASE_ERROR_RMS_DEG = 10.0

RANDOM_SEED = 42


def calculate_faulty_pattern(
    theta_values,
    failed_elements=None,
    phase_errors=None
):
    
    #Calculate the radiation pattern with optional faults.
    #failed_elements: Boolean array. True means the element has failed.
    #phase_errors: Random phase error for each element, in radians.


    if failed_elements is None:
        failed_elements = np.zeros(
            config.TOTAL_ELEMENTS,
            dtype=bool
        )

    if phase_errors is None:
        phase_errors = np.zeros(config.TOTAL_ELEMENTS)

    # Observation angles for phi = 0 degrees
    theta = np.radians(theta_values)
    phi = 0.0

    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)

    x_positions, y_positions = get_element_positions()

    # Phase due to propagation
    observation_phase = config.WAVE_NUMBER * (
        x_positions[:, None] * u[None, :]
        + y_positions[:, None] * v[None, :]
    )

    # Required phase for beam steering
    steering_phases = calculate_steering_phases()

    # Element excitation including phase errors
    excitation = np.exp(
        1j * (
            steering_phases
            + phase_errors
        )
    )

    # Disable failed elements
    excitation[failed_elements] = 0.0

    # Calculate combined electromagnetic field
    total_field = np.sum(
        excitation[:, None]
        * np.exp(1j * observation_phase),
        axis=0
    )

    # Calculate normalized element power pattern
    element_pattern = np.maximum(
        np.cos(theta),
        0.0
    ) ** config.ELEMENT_PATTERN_EXPONENT

    # Convert field magnitude into power
    power = np.abs(total_field) ** 2 * element_pattern

    return power


def create_fault_scenarios():
    #Generate reproducible fault conditions

    rng = np.random.default_rng(RANDOM_SEED)

    # Number of failed elements
    failed_count = int(
        round(
            config.TOTAL_ELEMENTS
            * FAILED_ELEMENT_PERCENT / 100
        )
    )

    failed_elements = np.zeros(
        config.TOTAL_ELEMENTS,
        dtype=bool
    )

    failed_indices = rng.choice(
        config.TOTAL_ELEMENTS,
        size=failed_count,
        replace=False
    )

    failed_elements[failed_indices] = True

    # Random Gaussian phase errors
    phase_errors = np.radians(
        rng.normal(
            loc=0.0,
            scale=PHASE_ERROR_RMS_DEG,
            size=config.TOTAL_ELEMENTS
        )
    )

    return failed_elements, phase_errors


def calculate_scenario_results(
    name,
    power,
    ideal_peak_power,
    theta_values
):
    #Extract performance metrics for a fault scenario

    peak_index = np.argmax(power)

    peak_theta = theta_values[peak_index]

    # Peak power relative to the ideal antenna peak
    relative_peak_db = 10 * np.log10(
        max(power[peak_index], 1e-15)
        / ideal_peak_power
    )

    pointing_error = abs(
        peak_theta - config.STEERING_THETA_DEG
    )

    return {
        "name": name,
        "peak_theta": peak_theta,
        "pointing_error": pointing_error,
        "relative_peak_db": relative_peak_db,
        "power": power
    }


def run_fault_simulation():
    #Run all fault scenarios and compare performance

    theta_values = np.linspace(0, 90, 721)

    # Generate fault conditions
    failed_elements, phase_errors = create_fault_scenarios()

    # Ideal antenna
    ideal_power = calculate_faulty_pattern(
        theta_values
    )

    ideal_peak_power = np.max(ideal_power)

    # Failed elements only
    failed_power = calculate_faulty_pattern(
        theta_values,
        failed_elements=failed_elements
    )

    # Phase errors only
    phase_power = calculate_faulty_pattern(
        theta_values,
        phase_errors=phase_errors
    )

    # Both faults combined
    combined_power = calculate_faulty_pattern(
        theta_values,
        failed_elements=failed_elements,
        phase_errors=phase_errors
    )

    scenarios = [
        calculate_scenario_results(
            "Ideal",
            ideal_power,
            ideal_peak_power,
            theta_values
        ),
        calculate_scenario_results(
            "Failed elements",
            failed_power,
            ideal_peak_power,
            theta_values
        ),
        calculate_scenario_results(
            "Phase errors",
            phase_power,
            ideal_peak_power,
            theta_values
        ),
        calculate_scenario_results(
            "Combined faults",
            combined_power,
            ideal_peak_power,
            theta_values
        )
    ]

    # Print summary of results

    print("=" * 65)
    print("SAatellite Phased-Array Fault Injection Simulation Results")
    print("=" * 65)

    print(f"Array configuration: {config.ELEMENTS_X} x {config.ELEMENTS_Y}")
    print(f"Total elements: {config.TOTAL_ELEMENTS}")

    print("-" * 65)

    print(
        f"Failed elements: {np.sum(failed_elements)} "
        f"({FAILED_ELEMENT_PERCENT:.1f}% target)"
    )

    print(
        f"Phase error RMS setting: "
        f"{PHASE_ERROR_RMS_DEG:.1f} degrees"
    )

    print("-" * 65)

    print(
        f"{'Scenario':<20}"
        f"{'Peak Angle':>14}"
        f"{'Pointing Error':>16}"
        f"{'Relative Peak':>15}"
    )

    print("-" * 65)

    for result in scenarios:

        print(
            f"{result['name']:<20}"
            f"{result['peak_theta']:>11.3f}°"
            f"{result['pointing_error']:>13.3f}°"
            f"{result['relative_peak_db']:>12.2f} dB"
        )

    print("=" * 65)

    # Plotting the radiation patterns

    plt.figure(figsize=(12, 7))

    for result in scenarios:

        # Reference all scenarios to the ideal peak
        power_db = 10 * np.log10(
            np.maximum(
                result["power"] / ideal_peak_power,
                1e-12
            )
        )

        plt.plot(
            theta_values,
            power_db,
            label=result["name"],
            linewidth=1.8
        )

    plt.axvline(
        config.STEERING_THETA_DEG,
        linestyle="--",
        label="Commanded beam direction"
    )

    plt.title(
        "Phased-Array Radiation Pattern Under Fault Conditions"
    )

    plt.xlabel("Theta Angle (degrees)")
    plt.ylabel("Relative Power (dB)")

    plt.ylim(-40, 2)
    plt.xlim(0, 90)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "fault_simulation_comparison.png",
        dpi=300
    )

    plt.show()

    print("Saved: fault_simulation_comparison.png")


if __name__ == "__main__":
    run_fault_simulation()
