
import numpy as np

from fault_simulation import (
    calculate_faulty_pattern,
    create_fault_scenarios,
    calculate_scenario_results
)


def test_fault_scenarios_are_repeatable():

    failed_1, phase_1 = create_fault_scenarios()
    failed_2, phase_2 = create_fault_scenarios()

    assert np.array_equal(failed_1, failed_2)
    assert np.array_equal(phase_1, phase_2)


def test_failed_element_count():

    failed, _ = create_fault_scenarios()

    assert np.sum(failed) == 26


def test_ideal_pattern_is_repeatable():

    theta = np.linspace(0, 90, 721)

    pattern_1 = calculate_faulty_pattern(theta)
    pattern_2 = calculate_faulty_pattern(theta)

    assert np.allclose(pattern_1, pattern_2)


def test_fault_pattern_has_valid_power():

    theta = np.linspace(0, 90, 721)

    failed, phase_errors = create_fault_scenarios()

    pattern = calculate_faulty_pattern(
        theta,
        failed_elements=failed,
        phase_errors=phase_errors
    )

    assert np.all(np.isfinite(pattern))
    assert np.all(pattern >= 0)
    assert np.max(pattern) > 0
