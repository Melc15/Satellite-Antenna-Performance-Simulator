
"""
Satellite Phased-Array Automated Verification System

Runs engineering verification checks against a set of demonstration acceptance criteria.

"""

import numpy as np

import antenna_config as config

from performance_analysis import calculate_performance

from fault_simulation import (calculate_faulty_pattern,create_fault_scenarios)


#Demonstration acceptance criteria for the phased-array antenna system.
REQUIREMENTS = {
    "ANT-001": {
        "description": "Beam pointing accuracy",
        "limit": 0.25,
        "unit": "degrees"
    },

    "ANT-002": {
        "description": "Half-power beamwidth",
        "minimum": 6.0,
        "maximum": 9.0,
        "unit": "degrees"
    },

    "ANT-003": {
        "description": "Peak sidelobe level",
        "limit": -10.0,
        "unit": "dB"
    },

    "ANT-004": {
        "description": "Degradation with 10% failed elements",
        "limit": 1.5,
        "unit": "dB"
    },

    "ANT-005": {
        "description": "Degradation with 10-degree RMS phase errors",
        "limit": 0.5,
        "unit": "dB"
    },

    "ANT-006": {
        "description": "Degradation with combined faults",
        "limit": 1.8,
        "unit": "dB"
    }
}


#Verification results will be stored in this list.
verification_results = []


def record_result(
    requirement_id,
    measured_value,
    limit_description,
    passed
):
    #Store an individual verification result.

    requirement = REQUIREMENTS[requirement_id]

    result = {
        "id": requirement_id,
        "description": requirement["description"],
        "measured": measured_value,
        "limit": limit_description,
        "status": "PASS" if passed else "FAIL"
    }

    verification_results.append(result)

    return result


def check_pointing_accuracy(performance):
    #Verify beam pointing accuracy.

    error = performance["pointing_error_deg"]
    limit = REQUIREMENTS["ANT-001"]["limit"]

    return record_result(
        "ANT-001",
        f"{error:.3f} degrees",
        f"<= {limit:.2f} degrees",
        error <= limit
    )


def check_beamwidth(performance):
    #Verify half-power beamwidth.

    width = performance["hpbw_deg"]

    minimum = REQUIREMENTS["ANT-002"]["minimum"]
    maximum = REQUIREMENTS["ANT-002"]["maximum"]

    passed = minimum <= width <= maximum

    return record_result(
        "ANT-002",
        f"{width:.3f} degrees",
        f"{minimum:.1f} to {maximum:.1f} degrees",
        passed
    )


def check_sidelobe_level(performance):
    #Verify peak sidelobe level.

    sidelobe = performance["peak_sll_db"]
    limit = REQUIREMENTS["ANT-003"]["limit"]

    passed = (
        sidelobe is not None
        and sidelobe <= limit
    )

    measured = (
        f"{sidelobe:.2f} dB"
        if sidelobe is not None
        else "Not detected"
    )

    return record_result(
        "ANT-003",
        measured,
        f"<= {limit:.1f} dB",
        passed
    )


def calculate_peak_degradation(
    faulty_power,
    ideal_peak_power
):
    #Calculate peak power degradation relative to ideal.

    faulty_peak = np.max(faulty_power)

    degradation = 10 * np.log10(
        max(faulty_peak, 1e-15) / ideal_peak_power
    )

    return abs(degradation)


def check_fault_degradation(
    requirement_id,
    faulty_power,
    ideal_peak_power
):
    #Verify the maximum permitted peak power degradation.

    degradation = calculate_peak_degradation(
        faulty_power,
        ideal_peak_power
    )

    limit = REQUIREMENTS[requirement_id]["limit"]

    passed = degradation <= limit

    return record_result(
        requirement_id,
        f"{degradation:.2f} dB",
        f"<= {limit:.2f} dB degradation",
        passed
    )


#MAIN verification function

def run_verification():

    verification_results.clear()

    print("=" * 75)
    print("Satellite Phased-Array Antenna Verification System")
    print("=" * 75)

    print("Reference design:")
    print(
        f"{config.ELEMENTS_X} x {config.ELEMENTS_Y} "
        f"array | {config.FREQUENCY_HZ / 1e9:.2f} GHz"
    )

    print("-" * 75)


    # Baseline performance tests

    print("Running baseline performance tests...")

    performance = calculate_performance()

    check_pointing_accuracy(performance)
    check_beamwidth(performance)
    check_sidelobe_level(performance)

    # Fault injection tests

    print("Running fault injection tests...")

    theta_values = np.linspace(0, 90, 721)

    failed_elements, phase_errors = create_fault_scenarios()

    ideal_power = calculate_faulty_pattern(
        theta_values
    )

    ideal_peak_power = np.max(ideal_power)

    failed_power = calculate_faulty_pattern(
        theta_values,
        failed_elements=failed_elements
    )

    phase_power = calculate_faulty_pattern(
        theta_values,
        phase_errors=phase_errors
    )

    combined_power = calculate_faulty_pattern(
        theta_values,
        failed_elements=failed_elements,
        phase_errors=phase_errors
    )

    check_fault_degradation(
        "ANT-004",
        failed_power,
        ideal_peak_power
    )

    check_fault_degradation(
        "ANT-005",
        phase_power,
        ideal_peak_power
    )

    check_fault_degradation(
        "ANT-006",
        combined_power,
        ideal_peak_power
    )

    # Verification report

    print()
    print("=" * 75)
    print("REQUIREMENTS VERIFICATION REPORT")
    print("=" * 75)

    print(
        f"{'ID':<12}"
        f"{'TEST DESCRIPTION':<48}"
        f"{'STATUS':>10}"
    )

    print("-" * 75)

    for result in verification_results:

        print(
            f"{result['id']:<12}"
            f"{result['description']:<48}"
            f"{result['status']:>10}"
        )

    print("-" * 75)

    passed_count = sum(
        result["status"] == "PASS"
        for result in verification_results
    )

    failed_count = len(verification_results) - passed_count

    total_count = len(verification_results)

    pass_rate = (
        passed_count / total_count * 100
        if total_count > 0
        else 0
    )

    print(f"Total tests: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Pass rate: {pass_rate:.1f}%")

    print("-" * 75)

    if failed_count == 0:
        print("OVERALL RESULT: PASS")
    else:
        print("OVERALL RESULT: FAIL")

    print("=" * 75)

    return verification_results


if __name__ == "__main__":
    run_verification()
