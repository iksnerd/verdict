"""Scorecard names that are not vN.N.N (audit 4)."""
from verdict.evaluation import bench

from efakes import write_cards


def test_a_scorecard_with_a_suffix_does_not_brick_the_listing(tmp_path):
    write_cards(tmp_path, {n: {"model": "m", "bits": 16, "budget": 1} for n in ("0.9.0", "0.10.2", "0.10.2-rc1", "notes")})
    assert [p.stem for p in bench.scorecards(tmp_path)] == ["v0.9.0", "v0.10.2"]
    assert bench.previous(tmp_path, "0.11.0").stem == "v0.10.2"
