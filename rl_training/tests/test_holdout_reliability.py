import json

import pytest

from rl_training.analyze_holdout_reliability import aggregate, wilson_interval


def test_wilson_interval_is_finite_for_all_successes():
    low, high = wilson_interval(90, 90)
    assert low == pytest.approx(.9591, abs=.0002)
    assert high == pytest.approx(1.)


def test_aggregate_counts_repeated_geometry_and_retains_failure(tmp_path):
    paths = []
    for index, success in enumerate((True, False)):
        path = tmp_path / f"summary_{index}.json"
        path.write_text(json.dumps({"size": 5, "cell_m": .75, "results": [{
            "maze_seed": 7, "success": success, "wall_contacts": int(not success),
            "geometry_collision": False, "model_sha256": "abc"}]}))
        paths.append(path)
    result = aggregate(paths)
    assert result["episodes"] == 2 and result["successes"] == 1
    assert result["unique_geometries"] == 1
    assert result["geometries_all_successful"] == 0
    assert result["geometry_all_success_wilson_95"][1] < 1
    assert result["wall_contacts"] == 1


def test_aggregate_reads_legacy_default_holdout_dimensions(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"results": [{"maze": "holdout_31001", "seed": 17, "success": True,
                                             "model_sha256": "abc"}]}))
    result = aggregate([path])
    assert "5x5:0.750:31001" in result["per_geometry"]
