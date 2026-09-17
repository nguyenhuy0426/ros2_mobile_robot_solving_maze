import numpy as np
import pytest
import torch

from rl_training.hybrid_navigation import LocalPolicy
from rl_training.oriented_navigation import TrackingPolicy
from rl_training.train_hybrid_multi import (
    ProceduralMaze,
    evaluator_maze_args,
    fit,
    load_dataset,
    manifest_maze,
    require_valid_evaluation,
    schedule_jobs,
    score,
)


def test_contact_failure_outranks_success_flag():
    assert score([{"success": True, "wall_contacts": 1, "contact_monitor_observed": True}]) == (-1, 0)
    assert score([{"success": True, "geometry_collision": True, "contact_monitor_observed": True}]) == (-1, 0)


def test_missing_contact_evidence_does_not_count_as_success():
    assert score([{"success": True}]) == (0, 0)
    assert score([{"success": True, "contact_monitor_observed": True}]) == (0, 1)


def test_broken_worker_cannot_be_used_for_model_selection():
    with pytest.raises(RuntimeError, match="repair simulation"):
        require_valid_evaluation([{"maze": "delta_1", "contact_monitor_observed": True,
                                   "reason": "no_motion_response"}])
    require_valid_evaluation([{"maze": "delta_1", "contact_monitor_observed": True, "reason": "timeout"}])
    with pytest.raises(RuntimeError, match="empty"):
        require_valid_evaluation([])


@pytest.mark.parametrize("reason", ["stale_sensors", "no_motion_response", "worker_timeout", "error: startup"])
def test_stale_rollout_is_excluded_from_learning(tmp_path, reason):
    np.savez(tmp_path / "demonstrations.npz", observations=np.ones((3, 38)), labels=np.ones((3, 2)))
    with pytest.raises(RuntimeError, match="No usable"):
        load_dataset([{"artifact_dir": str(tmp_path), "reason": reason}])


def test_worker_datasets_are_pooled_without_shape_changes(tmp_path):
    rows = []
    for i in range(2):
        d = tmp_path / str(i)
        d.mkdir()
        np.savez(d / "demonstrations.npz", observations=np.full((2, 38), i, dtype=np.float32),
                 labels=np.full((2, 2), i, dtype=np.float32))
        rows.append({"artifact_dir": str(d), "reason": "goal"})
    x, y = load_dataset(rows)
    assert x.shape == (4, 38) and y.shape == (4, 2)
    assert x[:, 0].tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize("policy,dim", [(LocalPolicy, 38), (TrackingPolicy, 42)])
def test_training_updates_weights_and_writes_optimizer(tmp_path, policy, dim):
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model = policy()
    initial = tmp_path / "initial.pt"
    output = tmp_path / "candidate.pt"
    torch.save(model.state_dict(), initial)
    x = np.full((32, dim), .5, dtype=np.float32)
    y = np.zeros((32, 2), dtype=np.float32)
    metrics = fit(initial, x, y, output, seed=2, epochs=2)
    assert metrics["gazebo_samples"] == 32
    weights = torch.load(output, weights_only=True)
    assert any(not torch.equal(v, weights[k]) for k, v in model.state_dict().items())
    assert output.with_suffix(".optimizer.pt").exists()


def test_separate_networks_never_mix_incompatible_observations(tmp_path):
    rows = []
    for dim in (38, 42):
        folder = tmp_path / str(dim)
        folder.mkdir()
        np.savez(folder / "demonstrations.npz", observations=np.ones((2, dim), dtype=np.float32),
                 labels=np.zeros((2, 2), dtype=np.float32))
        rows.append({"artifact_dir": str(folder), "reason": "goal", "observation_dim": dim})
    assert load_dataset(rows)[0].shape == (2, 38)
    assert load_dataset(rows, input_dim=42)[0].shape == (2, 42)


@pytest.mark.parametrize("workers", [1, 2, 3, 4])
def test_scheduling_preserves_episode_indices_and_spreads_long_jobs(workers):
    names = ["delta_1", "delta_2", "ortho_1", "ortho_2", "sigma_1", "sigma_2", "sigma_3", "sigma_5"]
    jobs = schedule_jobs(names, workers)
    assert sorted(i for bucket in jobs for i in bucket) == list(range(len(names)))
    if workers == 4:
        assert all(sum(names[i].startswith("sigma") for i in bucket) == 1 for bucket in jobs)


def test_procedural_maze_has_reproducible_evaluator_and_manifest_identity():
    maze = ProceduralMaze(seed=41001, size=6, cell=.70)
    assert maze.name == "holdout_6x6_70cm_41001"
    assert evaluator_maze_args(maze) == [
        "--holdout-seed", "41001", "--holdout-size", "6", "--holdout-cell", "0.7"]
    assert manifest_maze(maze) == {
        "name": maze.name, "seed": 41001, "size": 6, "cell": .70}


def test_scheduler_accepts_registry_and_procedural_mazes_together():
    mazes = ["sigma_1", ProceduralMaze(41001, 6, .7), "delta_1"]
    jobs = schedule_jobs(mazes, workers=2)
    assert sorted(i for bucket in jobs for i in bucket) == [0, 1, 2]
