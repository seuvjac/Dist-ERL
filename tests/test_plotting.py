"""Regression tests for paper plotting helpers."""

import numpy as np

from scripts.plot_fedrl_heterogeneous import (
    _align_runs_to_first_evaluation,
    _smooth_nan,
)


def test_smoothing_preserves_support_and_endpoints():
    values = np.array([np.nan, np.nan, 1.0, 9.0, 5.0, 7.0, np.nan])

    smoothed = _smooth_nan(values, 5)

    assert np.isnan(smoothed[:2]).all()
    assert np.isnan(smoothed[-1])
    assert smoothed[2] == values[2]
    assert smoothed[5] == values[5]
    assert smoothed[3] == np.mean(values[2:5])


def test_align_runs_only_shifts_x_axis():
    run = {
        'x': np.array([120.0, 180.0, 260.0]),
        'y': np.array([2.0, 3.0, 5.0]),
        'y_std': np.array([0.2, 0.3, 0.5]),
    }

    aligned = _align_runs_to_first_evaluation([run])

    np.testing.assert_array_equal(aligned[0]['x'], [0.0, 60.0, 140.0])
    np.testing.assert_array_equal(aligned[0]['y'], run['y'])
    np.testing.assert_array_equal(aligned[0]['y_std'], run['y_std'])
    np.testing.assert_array_equal(run['x'], [120.0, 180.0, 260.0])
