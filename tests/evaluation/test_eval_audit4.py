"""Scorecard names that are not vN.N.N (audit 4)."""
import json

from verdict.evaluation import bench


def test_a_scorecard_with_a_suffix_does_not_brick_the_listing(tmp_path):
    for n in ("v0.9.0", "v0.10.2", "v0.10.2-rc1", "vnotes"):
        (tmp_path / f"{n}.json").write_text(json.dumps({"model": "m", "bits": 16, "budget": 1}))
    assert [p.stem for p in bench.scorecards(tmp_path)] == ["v0.9.0", "v0.10.2"]
    assert bench.previous(tmp_path, "0.11.0").stem == "v0.10.2"
