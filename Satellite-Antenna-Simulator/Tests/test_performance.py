
from performance_analysis import calculate_performance


def test_pointing_error_is_within_expected_range():

    results = calculate_performance()

    assert results["pointing_error_deg"] <= 0.25


def test_beamwidth_is_positive():

    results = calculate_performance()

    assert results["hpbw_deg"] > 0


def test_sidelobe_level_is_detected():

    results = calculate_performance()

    assert results["peak_sll_db"] is not None
    assert results["peak_sll_db"] < 0
