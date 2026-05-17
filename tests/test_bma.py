"""BMA orchestrator tests."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from helios_fusion.bma import BMAOrchestrator, compute_skill_weights, renormalize_weights
from helios_fusion.types import ModelOutput


def test_equal_weight_recovers_naive_mean(
    synthetic_model_outputs: dict[str, list[ModelOutput]],
) -> None:
    """Equal weights ⇒ fused value equals the arithmetic mean of inputs."""
    model_ids = list(synthetic_model_outputs.keys())
    weights = {m: 1.0 for m in model_ids}
    bma = BMAOrchestrator(weights=weights, prediction_target="test")

    outputs_at_t0 = [synthetic_model_outputs[m][0] for m in model_ids]
    fused = bma.fuse(outputs_at_t0)

    expected = np.mean([o.value for o in outputs_at_t0])
    assert fused.value == pytest.approx(float(expected), rel=1e-9)
    assert fused.lineage[0].transformation_type == "bma_fuse"
    # Weights in lineage should sum to 1.
    used = fused.lineage[0].parameters["weights_used"]
    assert sum(used.values()) == pytest.approx(1.0, abs=1e-9)


def test_skill_weighted_favours_well_calibrated(
    synthetic_model_outputs: dict[str, list[ModelOutput]],
    synthetic_truth: np.ndarray,
) -> None:
    """update_weights should assign higher weight to the well-calibrated model."""
    bma = BMAOrchestrator(weight_policy="hss_clipped")

    # Build a verification window: pair every (record, observed) from the
    # synthetic stream for the first 200 timesteps.
    window: list[tuple[ModelOutput, float]] = []
    for model_id, records in synthetic_model_outputs.items():
        for i, rec in enumerate(records[:200]):
            window.append((rec, float(synthetic_truth[i])))

    bma.update_weights(window)
    weights = bma.weights
    assert weights is not None
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    # Well-calibrated model should rank in the top half of the 5 streams.
    sorted_ids = sorted(weights, key=lambda m: weights[m], reverse=True)
    top_half = sorted_ids[: len(sorted_ids) // 2 + 1]
    assert "well_calibrated" in top_half


def test_missing_model_renormalises() -> None:
    """If a configured model is missing at fuse time, remaining weights renormalise."""
    bma = BMAOrchestrator(
        weights={"A": 0.5, "B": 0.3, "C": 0.2},
        prediction_target="missing_test",
    )
    t = datetime(2024, 1, 1, tzinfo=UTC)
    outputs = [
        ModelOutput(id="a1", model_id="A", timestamp=t, value=0.8, value_units="probability"),
        ModelOutput(id="b1", model_id="B", timestamp=t, value=0.4, value_units="probability"),
        # C is missing
    ]
    fused = bma.fuse(outputs)

    # Renormalised: A_eff = 0.5 / 0.8 = 0.625; B_eff = 0.3 / 0.8 = 0.375
    expected = 0.625 * 0.8 + 0.375 * 0.4
    assert fused.value == pytest.approx(expected, rel=1e-9)
    assert fused.lineage[0].parameters["excluded_models"] == ["C"]
    used = fused.lineage[0].parameters["weights_used"]
    assert "C" not in used
    assert sum(used.values()) == pytest.approx(1.0, abs=1e-9)


def test_fuse_raises_when_no_weights_set() -> None:
    bma = BMAOrchestrator()
    t = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="weights not set"):
        bma.fuse(
            [
                ModelOutput(
                    id="x", model_id="A", timestamp=t, value=0.5, value_units="probability"
                ),
            ]
        )


def test_fuse_raises_on_mismatched_timestamps() -> None:
    bma = BMAOrchestrator(weights={"A": 1.0, "B": 1.0})
    t1 = datetime(2024, 1, 1, tzinfo=UTC)
    t2 = datetime(2024, 1, 2, tzinfo=UTC)
    with pytest.raises(ValueError, match="share a timestamp"):
        bma.fuse(
            [
                ModelOutput(
                    id="x", model_id="A", timestamp=t1, value=0.5, value_units="probability"
                ),
                ModelOutput(
                    id="y", model_id="B", timestamp=t2, value=0.7, value_units="probability"
                ),
            ]
        )


def test_fuse_raises_when_no_overlap_with_configured_weights() -> None:
    bma = BMAOrchestrator(weights={"A": 1.0})
    t = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="none of the supplied outputs"):
        bma.fuse(
            [
                ModelOutput(
                    id="x", model_id="Z", timestamp=t, value=0.5, value_units="probability"
                ),
            ]
        )


def test_update_weights_empty_window_raises() -> None:
    bma = BMAOrchestrator()
    with pytest.raises(ValueError, match="must not be empty"):
        bma.update_weights([])


def test_compute_skill_weights_softmax_concentrates_on_best() -> None:
    skill = {"a": 0.9, "b": 0.5, "c": 0.1}
    w = compute_skill_weights(skill, policy="softmax", softmax_beta=10.0)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert w["a"] > w["b"] > w["c"]
    assert w["a"] > 0.8  # heavily concentrated


def test_compute_skill_weights_linear_with_negative_scores() -> None:
    skill = {"a": 0.5, "b": -0.2}
    w = compute_skill_weights(skill, policy="linear")
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert w["a"] > w["b"]
    # Negative score is clipped to 0 then mixed with epsilon.
    assert w["b"] < 0.01


def test_compute_skill_weights_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="unknown weight policy"):
        compute_skill_weights({"a": 0.5}, policy="bogus")  # type: ignore[arg-type]


def test_compute_skill_weights_empty_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_skill_weights({})


def test_renormalize_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        renormalize_weights({"a": -0.1, "b": 0.5})


def test_renormalize_rejects_zero_sum() -> None:
    with pytest.raises(ValueError, match="sum to zero"):
        renormalize_weights({"a": 0.0, "b": 0.0})


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=5),
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=50)
def test_renormalize_weights_sum_to_one(raw: dict[str, float]) -> None:
    w = renormalize_weights(raw)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0.0 for v in w.values())


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=5),
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    )
)
@settings(max_examples=50)
def test_compute_skill_weights_sum_to_one(skill: dict[str, float]) -> None:
    w = compute_skill_weights(skill)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)


def test_duplicate_model_in_outputs_emits_warning(caplog) -> None:
    bma = BMAOrchestrator(weights={"A": 1.0})
    t = datetime(2024, 1, 1, tzinfo=UTC)
    outputs = [
        ModelOutput(id="a1", model_id="A", timestamp=t, value=0.3, value_units="probability"),
        ModelOutput(id="a2", model_id="A", timestamp=t, value=0.7, value_units="probability"),
    ]
    import logging

    with caplog.at_level(logging.WARNING):
        fused = bma.fuse(outputs)
    assert "duplicate" in caplog.text.lower()
    # Implementation uses last entry.
    assert fused.value == pytest.approx(0.7, abs=1e-9)


def test_update_weights_degenerate_falls_back_to_zero_skill() -> None:
    """All-zeros prediction stream should fall back to skill=0 without crashing."""
    bma = BMAOrchestrator()
    t = datetime(2024, 1, 1, tzinfo=UTC)
    window = [
        (
            ModelOutput(
                id=f"a{i}", model_id="A", timestamp=t, value=0.0, value_units="probability"
            ),
            float(i % 2),
        )
        for i in range(10)
    ]
    bma.update_weights(window)
    weights = bma.weights
    assert weights is not None
    assert "A" in weights
    assert weights["A"] == pytest.approx(1.0, abs=1e-9)  # only model present
