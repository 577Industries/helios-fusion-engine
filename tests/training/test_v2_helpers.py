"""Unit tests for v2 helper functions (component-id, variant inference, etc.)."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

loader_module = importlib.import_module("helios_fusion.training.load_table_3_1")
archive_module = importlib.import_module("helios_fusion.training.swpc_sep_archive")


def test_component_id_from_spec_umasep() -> None:
    """UMASEP entries have explicit energy directories."""
    out = loader_module._component_id_from_spec("UMASEP", ("v2_0",), "10MeV")
    assert out == "UMASEP/v2_0/10MeV"


def test_component_id_from_spec_empty_energy() -> None:
    """Empty energy renders as ``noE``."""
    out = loader_module._component_id_from_spec("SEPSTER", ("Parker",), "")
    assert out == "SEPSTER/Parker/noE"


def test_component_id_from_spec_multi_variant_chain() -> None:
    """Multi-segment variant chains get joined with underscores."""
    out = loader_module._component_id_from_spec("SAWS_ASPECS", ("1.X", "Nowcasts", "Profile"), "")
    assert out == "SAWS_ASPECS/1.X_Nowcasts_Profile/noE"


def test_component_id_from_spec_no_variants() -> None:
    """Variants-less models render as ``noVariant``."""
    out = loader_module._component_id_from_spec("FOO", (), "10MeV")
    assert out == "FOO/noVariant/10MeV"


def test_infer_variants_from_url() -> None:
    """Real URL paths should yield the correct variant chain."""
    by_name = {
        "UMASEP": [("v3_X",), ("v2_0",)],
        "SAWS_ASPECS": [("1.X", "Nowcasts", "Profile"), ("1.X", "Forecasts", "Probability")],
    }
    prefix = "/iswa_data_tree/model/heliosphere/sep_scoreboard"
    url = f"https://iswa.ccmc.gsfc.nasa.gov{prefix}/UMASEP/v2_0/10MeV/2017/09/file.json"
    out = loader_module._infer_variants("UMASEP", url, by_name, prefix=prefix)
    assert out == ("v2_0",)


def test_infer_variants_falls_back_to_first_when_url_missing() -> None:
    """When the URL doesn't include the registry prefix, fall back to the
    first registered variant chain for that model."""
    by_name = {"UMASEP": [("v3_X",), ("v2_0",)]}
    out = loader_module._infer_variants(
        "UMASEP", "", by_name, prefix="/iswa_data_tree/model/heliosphere/sep_scoreboard"
    )
    assert out == ("v3_X",)


def test_infer_variants_unknown_model_returns_empty() -> None:
    """Models not in the registry yield an empty variant tuple."""
    out = loader_module._infer_variants(
        "FOO", "", {}, prefix="/iswa_data_tree/model/heliosphere/sep_scoreboard"
    )
    assert out == ()


def test_infer_variants_multi_segment_chain() -> None:
    """SAWS_ASPECS-style chains are recognised end-to-end."""
    by_name = {
        "SAWS_ASPECS": [
            ("1.X", "Nowcasts", "Profile"),
            ("1.X", "Forecasts", "Probability"),
        ],
    }
    prefix = "/iswa_data_tree/model/heliosphere/sep_scoreboard"
    url = (
        f"https://iswa.ccmc.gsfc.nasa.gov{prefix}/SAWS_ASPECS/"
        "1.X/Nowcasts/Profile/2017/09/file.json"
    )
    out = loader_module._infer_variants("SAWS_ASPECS", url, by_name, prefix=prefix)
    assert out == ("1.X", "Nowcasts", "Profile")


def test_infer_energy_umasep_url() -> None:
    """UMASEP URLs encode the energy directory; we extract it."""
    url = (
        "https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/model/heliosphere/"
        "sep_scoreboard/UMASEP/v2_0/100MeV/2017/09/file.json"
    )
    out = loader_module._infer_energy(url, "UMASEP", ("v2_0",))
    assert out == "100MeV"


def test_infer_energy_returns_empty_for_non_umasep() -> None:
    """Non-UMASEP models always use empty-energy paths."""
    url = "https://example.com/SEPSTER/Parker/2017/09/file.json"
    assert loader_module._infer_energy(url, "SEPSTER", ("Parker",)) == ""


def test_infer_energy_returns_empty_for_empty_url() -> None:
    """No URL → no energy info."""
    assert loader_module._infer_energy("", "UMASEP", ("v2_0",)) == ""


def test_kp_synthesis_centred_on_onset() -> None:
    """Synthesised Kp series peaks near the onset time."""
    import numpy as np

    from helios_fusion.training.load_table_3_1 import TRAINING_EVENTS

    event = TRAINING_EVENTS[0]
    rng = np.random.default_rng(42)
    grid = [event.window_start, event.onset, event.window_end]
    kp = loader_module._synthesize_kp_profile(event, grid, rng=rng)
    # Onset bin should have the highest Kp (centre of the bump).
    assert kp[1] > kp[0]
    assert kp[1] > kp[2]


def test_synthesize_truth_marks_window_near_onset() -> None:
    """Synthesised truth label is 1 within +/-24h of onset AND high Kp."""
    from datetime import timedelta

    import numpy as np

    from helios_fusion.training.load_table_3_1 import TRAINING_EVENTS

    event = TRAINING_EVENTS[0]
    rng = np.random.default_rng(42)
    grid = [event.onset + timedelta(hours=i - 12) for i in range(24)]
    # High Kp uniformly across the grid.
    kp = np.full(len(grid), 7.0)
    truth = loader_module._synthesize_truth(grid, event, kp, rng=rng)
    # At least some bins should be 1.
    assert truth.sum() > 0


def test_swpc_archive_event_truth_labels_uses_default_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If no archive_df is supplied to event_truth_labels, it fetches
    via fetch_sep_event_list (which we stub to a fixture)."""
    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "swpc_archive" / "seps_trimmed_v2.html"
    )
    html = fixture_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        archive_module,
        "_http_get_archive",
        lambda *, timeout: html,
    )
    # Use a temp cache_dir so we don't write to the real cache.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        labels = archive_module.event_truth_labels(
            "sep_2017",
            (datetime(2017, 9, 1, tzinfo=UTC), datetime(2017, 9, 16, tzinfo=UTC)),
            cache_dir=Path(td),
            cadence_hours=6.0,
        )
        assert not labels.empty


def test_swpc_archive_force_refresh_skips_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """force_refresh=True ignores any existing cached file."""
    import tempfile

    counter = {"fetch": 0}

    def fake_fetch(*, timeout: float) -> str:
        counter["fetch"] += 1
        return "<html><body><table></table></body></html>"

    monkeypatch.setattr(archive_module, "_http_get_archive", fake_fetch)
    with tempfile.TemporaryDirectory() as td:
        # First call: fetch and cache.
        archive_module.fetch_sep_event_list(cache_dir=Path(td))
        # Second call: cache hit, no fetch.
        archive_module.fetch_sep_event_list(cache_dir=Path(td))
        assert counter["fetch"] == 1
        # Third call with force_refresh: fetch again.
        archive_module.fetch_sep_event_list(cache_dir=Path(td), force_refresh=True)
        assert counter["fetch"] == 2


def test_swpc_archive_parser_skips_event_rows_before_year_header() -> None:
    """Event rows that appear before any <strong>YYYY</strong> header are
    dropped (no year context)."""
    html = """
    <table>
    <tr><td>Jul 14/1045<td>Jul 15/1230<td align = right>50<td></tr>
    <tr><td><td><td><td><strong>2024</strong><td></tr>
    <tr><td>Aug 01/0500<td>Aug 02/0500<td align = right>100<td></tr>
    </table>
    """
    df = archive_module.fetch_sep_event_list(html=html)
    # Only the post-year-header row should be parsed.
    assert len(df) == 1
    assert df.iloc[0]["year"] == 2024


def test_swpc_archive_event_truth_labels_handles_naive_datetimes() -> None:
    """event_truth_labels accepts naive datetimes (treats as UTC)."""
    import pandas as pd

    archive_df = pd.DataFrame(
        {
            "start_utc": pd.to_datetime(["2024-01-01 12:00:00"], utc=True),
            "max_utc": pd.to_datetime(["2024-01-01 18:00:00"], utc=True),
            "peak_flux_pfu": [100.0],
            "year": [2024],
            "event_label": ["test"],
        }
    )
    labels = archive_module.event_truth_labels(
        "test",
        (datetime(2024, 1, 1), datetime(2024, 1, 2)),  # naive
        archive_df=archive_df,
        cadence_hours=1.0,
    )
    assert int(labels["observed"].sum()) > 0
