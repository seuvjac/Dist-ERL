"""Regression tests for paired paper-level statistics."""

import numpy as np

from scripts.test_fedrl_significance import _rank_biserial, _wilcoxon


def test_rank_biserial_has_expected_direction():
    assert _rank_biserial([3.0, 2.0, 1.0]) == 1.0
    assert _rank_biserial([-3.0, -2.0, -1.0]) == -1.0
    assert _rank_biserial([0.0, 0.0]) == 0.0


def test_five_pairs_cannot_reach_two_sided_exact_five_percent():
    _, p_value = _wilcoxon(np.ones(5), 'two-sided')

    assert np.isclose(p_value, 0.0625)
