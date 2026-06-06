"""Fast, deterministic tests for the data-fraction scaling experiment.

These tests exercise the pure subsample helper only -- no real feature files,
no model training -- so they run in well under a second.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_data_scale import subsample_stage


def _make_samples(n: int, class_id: int = 0) -> list:
    return [{"id": f"vid{i:04d}", "class_id": class_id} for i in range(n)]


def _ids(samples: list) -> list:
    return [s["id"] for s in samples]


# ── backbone-independence / determinism ───────────────────────────────────────
def test_selection_is_backbone_independent():
    """Selection depends only on (seed, fraction, stage_id) + sample identity.

    The helper never receives a feature-dir, so the exact same chosen ids must
    come back no matter what the caller's feature backbone is. We emulate two
    backbones by shuffling the incoming list order (as different pipelines might
    produce) and confirm the chosen set is identical.
    """
    samples_a = _make_samples(40)
    # A "different backbone" might hand us the same samples in a different order.
    samples_b = list(reversed(samples_a))

    chosen_a = set(_ids(subsample_stage(samples_a, 0.5, seed=1, stage_id=3)))
    chosen_b = set(_ids(subsample_stage(samples_b, 0.5, seed=1, stage_id=3)))

    assert chosen_a == chosen_b


def test_selection_is_deterministic_across_calls():
    samples = _make_samples(40)
    first = _ids(subsample_stage(samples, 0.5, seed=2, stage_id=4))
    second = _ids(subsample_stage(samples, 0.5, seed=2, stage_id=4))
    assert first == second


def test_selection_varies_with_seed_and_stage():
    samples = _make_samples(40)
    base = set(_ids(subsample_stage(samples, 0.5, seed=0, stage_id=1)))
    other_seed = set(_ids(subsample_stage(samples, 0.5, seed=1, stage_id=1)))
    other_stage = set(_ids(subsample_stage(samples, 0.5, seed=0, stage_id=2)))
    # Not strictly required, but the RNG keying should make these differ.
    assert base != other_seed
    assert base != other_stage


# ── fraction sizing ───────────────────────────────────────────────────────────
def test_fraction_one_keeps_all():
    samples = _make_samples(37)
    out = subsample_stage(samples, 1.0, seed=0, stage_id=1)
    assert len(out) == 37
    assert set(_ids(out)) == set(_ids(samples))


def test_fraction_half_keeps_about_half():
    samples = _make_samples(40)
    out = subsample_stage(samples, 0.5, seed=0, stage_id=1)
    assert abs(len(out) - 20) <= 1


def test_fraction_quarter_keeps_about_quarter():
    samples = _make_samples(40)
    out = subsample_stage(samples, 0.25, seed=0, stage_id=1)
    assert abs(len(out) - 10) <= 1


def test_never_returns_zero_for_nonempty_stage():
    samples = _make_samples(3)
    # Tiny fraction on a small stage must still keep at least one sample.
    out = subsample_stage(samples, 0.01, seed=0, stage_id=1)
    assert len(out) >= 1


def test_empty_stage_returns_empty():
    assert subsample_stage([], 0.5, seed=0, stage_id=1) == []


def test_chosen_are_subset_of_input():
    samples = _make_samples(50)
    out = subsample_stage(samples, 0.3, seed=7, stage_id=5)
    assert set(_ids(out)).issubset(set(_ids(samples)))
    # No duplicates.
    assert len(_ids(out)) == len(set(_ids(out)))


def test_fraction_subset_is_nested_subset_of_larger_fraction_when_ge():
    """fraction>=1.0 returns everything, so half is a subset of full."""
    samples = _make_samples(40)
    full = set(_ids(subsample_stage(samples, 1.0, seed=3, stage_id=2)))
    half = set(_ids(subsample_stage(samples, 0.5, seed=3, stage_id=2)))
    assert half.issubset(full)
