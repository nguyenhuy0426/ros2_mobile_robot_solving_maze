"""Regressions for mixed trainers, observation aliasing and warm starts."""

from argparse import Namespace
import subprocess
import sys
import threading
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import SAC

from rl_training import config as C
from rl_training.exploration_memory import VisitMemory
from rl_training.training_session import training_lock


class TinyExploreEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (C.EXPL_STATE_DIM,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        obs = np.zeros(C.EXPL_STATE_DIM, np.float32)
        return obs, {}

    def step(self, action):
        self.steps += 1
        obs = np.zeros(C.EXPL_STATE_DIM, np.float32)
        return obs, 0.0, False, self.steps >= 4, {}

    def pre_send(self, action):
        pass

    def hold(self):
        pass


def test_same_current_state_different_history_is_distinguishable():
    left, right = (VisitMemory(TinyExploreEnv()) for _ in range(2))
    left.reset()
    right.reset()
    obs = np.zeros(C.EXPL_STATE_DIM, np.float32)
    obs[-6:-4] = [-0.9, 0]
    left.observation(obs)
    obs[-6:-4] = [0.9, 0]
    right.observation(obs)
    obs[-6:-4] = [0, 0]
    a, b = left.observation(obs), right.observation(obs)
    np.testing.assert_array_equal(a[:150], b[:150])
    assert not np.array_equal(a[150:], b[150:])
    assert left.observation_space.contains(a)
    assert a.shape == (175,)
    reset, _ = left.reset()
    assert reset[150:].sum() == 1
    assert a[150:].sum() == 2  # returned observations do not alias the bitmap


def test_lock_excludes_other_process_and_releases_on_error(tmp_path):
    path = tmp_path / 'world.lock'
    code = ('from pathlib import Path; '
            'from rl_training.training_session import training_lock; '
            'ctx = training_lock(Path(__import__("sys").argv[1])); '
            'ctx.__enter__(); ctx.__exit__(None, None, None)')
    with pytest.raises(ValueError):
        with training_lock(path):
            result = subprocess.run([sys.executable, '-c', code, str(path)],
                                    capture_output=True, text=True)
            assert result.returncode != 0
            assert 'Another multi-robot trainer' in result.stderr
            raise ValueError('simulated trainer crash')
    assert subprocess.run([sys.executable, '-c', code, str(path)]).returncode == 0


def test_stale_sensor_step_stops_instead_of_returning_transition():
    from rl_training.explore_env import GazeboExploreEnv
    env = GazeboExploreEnv.__new__(GazeboExploreEnv)
    env.robot_id = 2
    env._lock = threading.Lock()
    env._scan_seq = env._pose_seq = env._stale_steps = 0
    env._send_seq0 = None
    env._stop_wheels = Mock()
    with pytest.raises(RuntimeError, match='no fresh scan'):
        env._wait_fresh_step(timeout=0)
    env._stop_wheels.assert_called_once()


def test_analyzer_does_not_infer_policy_trend_from_mixed_counters():
    from rl_training.analyze_explore_log import verdict
    result = '\n'.join(verdict([{'timesteps': 100}, {'timesteps': 13}]))
    assert 'ownership' in result
    assert 'More timesteps' not in result


def test_actual_sac_warm_start_adds_only_requested_steps(tmp_path, monkeypatch):
    from rl_training import train_explore_multi as trainer
    # Small real SAC model: tests SB3 counter semantics, not just mocked calls.
    env = TinyExploreEnv()
    model = SAC('MlpPolicy', env, learning_starts=100, buffer_size=100,
                policy_kwargs={'net_arch': [8]}, device='cpu')
    model.learn(12)
    checkpoint = tmp_path / 'seed.zip'
    model.save(checkpoint)
    monkeypatch.setattr(trainer, '_make_env', lambda **kw: TinyExploreEnv())
    args = Namespace(run_dir=tmp_path / 'run', visit_memory=False,
                     seed=0, stall_limit=None, roam_bonus=None,
                     n_robots=1, bootstrap_scripted=0,
                     load=str(checkpoint), load_buffer=None,
                     gradient_steps=1,
                     ckpt_every=100, reset_elites=False, steps=8)
    trainer._train(args, ['ortho_1'])
    final = SAC.load(args.run_dir / 'checkpoints/sac_explore_multi_final.zip')
    assert final.num_timesteps == 20
    assert final.learning_starts == 12 + C.SACW_LEARN_STARTS
    assert (args.run_dir / 'run.json').exists()
    with pytest.raises(FileExistsError):
        trainer._train(args, ['ortho_1'])


def test_memory_survives_vector_terminal_observation_before_reset():
    from rl_training.concurrent_vec_env import ConcurrentVecEnv
    env = ConcurrentVecEnv([lambda: VisitMemory(TinyExploreEnv())])
    try:
        env.reset()
        for _ in range(4):
            obs, _, done, infos = env.step(np.zeros((1, 2), np.float32))
        assert done[0]
        assert infos[0]['terminal_observation'].shape == (175,)
        assert obs.shape == (1, 175)
        assert obs[0, 150:].sum() == 1
    finally:
        env.close()


def test_vector_stops_every_robot_before_updates_and_on_sensor_error():
    from rl_training.concurrent_vec_env import ConcurrentVecEnv
    robots = [TinyExploreEnv(), TinyExploreEnv()]
    for robot in robots:
        robot.hold = Mock()
    env = ConcurrentVecEnv([lambda r=r: r for r in robots])
    try:
        env.reset()
        env.step(np.zeros((2, 2), np.float32))
        # Each robot is held twice: once as its own control period closes,
        # once in the sweep that guards the reset/update window.
        for robot in robots:
            assert robot.hold.call_count == 2
        robots[0].step = Mock(side_effect=RuntimeError('sensor timeout'))
        with pytest.raises(RuntimeError):
            env.step(np.zeros((2, 2), np.float32))
        # A mid-loop sensor failure must still stop every robot, including
        # the ones whose step() never ran.
        assert all(r.hold.call_count >= 3 for r in robots)
    finally:
        env.close()


def test_sac_learns_with_memory_and_update_ratio_per_transition():
    from rl_training.concurrent_vec_env import ConcurrentVecEnv
    env = ConcurrentVecEnv([lambda: VisitMemory(TinyExploreEnv())
                            for _ in range(2)])
    try:
        model = SAC('MlpPolicy', env, learning_starts=0, buffer_size=100,
                    batch_size=4, gradient_steps=-1,
                    policy_kwargs={'net_arch': [8]}, device='cpu')
        model.learn(8)
        assert model._n_updates == 8
        assert model.replay_buffer.observation_space.shape == (175,)
    finally:
        env.close()


def test_explore_default_stall_grace_is_long_enough_for_one_controlled_turn():
    assert C.EXPL_STALL_LIMIT == 60


def test_safety_throttle_keeps_speed_in_open_space_and_slows_wall_approach():
    from rl_training.explore_env import safety_action_scale
    assert safety_action_scale(0.20, 0.08) == 1.0
    assert safety_action_scale(0.0, 0.08) == 0.35
    assert 0.35 < safety_action_scale(0.04, 0.08) < 1.0


def test_warm_start_with_replay_buffer_updates_from_the_first_step(tmp_path,
                                                                   monkeypatch):
    """A resumed campaign must not re-enter a random-action warmup.

    Warm starting from a .zip alone sets learning_starts to
    num_timesteps + SACW_LEARN_STARTS, so an attempt that dies before that
    threshold performs ZERO gradient updates and acts uniformly at random for
    its whole life -- which is exactly how a 13-robot campaign burned tens of
    hours without learning anything. Carrying the replay buffer across the
    restart removes the reason for that warmup.
    """
    from rl_training import train_explore_multi as trainer
    env = TinyExploreEnv()
    model = SAC('MlpPolicy', env, learning_starts=0, buffer_size=100,
                batch_size=4, policy_kwargs={'net_arch': [8]}, device='cpu')
    model.learn(12)
    checkpoint, buffer = tmp_path / 'seed.zip', tmp_path / 'seed.pkl'
    model.save(checkpoint)
    model.save_replay_buffer(buffer)
    monkeypatch.setattr(trainer, '_make_env', lambda **kw: TinyExploreEnv())
    args = Namespace(run_dir=tmp_path / 'run', visit_memory=False,
                     seed=0, stall_limit=None, roam_bonus=None,
                     n_robots=1, bootstrap_scripted=0,
                     load=str(checkpoint), load_buffer=str(buffer),
                     gradient_steps=1, ckpt_every=100, reset_elites=False,
                     steps=8)
    trainer._train(args, ['ortho_1'])
    final = SAC.load(args.run_dir / 'checkpoints/sac_explore_multi_final.zip')
    assert final.learning_starts == 12          # no fresh warmup
    assert (args.run_dir / 'checkpoints/replay_buffer.pkl').exists()


def test_warm_start_without_buffer_still_refills_before_updating(tmp_path,
                                                                 monkeypatch):
    from rl_training import train_explore_multi as trainer
    env = TinyExploreEnv()
    model = SAC('MlpPolicy', env, learning_starts=100, buffer_size=100,
                policy_kwargs={'net_arch': [8]}, device='cpu')
    model.learn(12)
    checkpoint = tmp_path / 'seed.zip'
    model.save(checkpoint)
    monkeypatch.setattr(trainer, '_make_env', lambda **kw: TinyExploreEnv())
    args = Namespace(run_dir=tmp_path / 'run', visit_memory=False,
                     seed=0, stall_limit=None, roam_bonus=None,
                     n_robots=1, bootstrap_scripted=0,
                     load=str(checkpoint), load_buffer=None,
                     gradient_steps=1, ckpt_every=100, reset_elites=False,
                     steps=8)
    trainer._train(args, ['ortho_1'])
    final = SAC.load(args.run_dir / 'checkpoints/sac_explore_multi_final.zip')
    assert final.learning_starts == 12 + C.SACW_LEARN_STARTS


def test_maze_pool_is_partitioned_disjointly_across_the_fleet():
    """Every registry maze must be trained on, and never by two robots at once.

    The fleet is capped at 4 robots because gpu_lidar rendering is
    single-threaded, but the policy still has to generalise over all 13
    mazes. Dealing the pool round-robin gives each robot a group it cycles
    through per episode; the groups must stay DISJOINT because two robots
    sent to the same maze would spawn on top of each other in the combined
    world, and EXHAUSTIVE because a maze in no group is never seen.
    """
    from rl_training.train_explore_multi import _partition_mazes
    import rl_training.maze_registry as MR

    pool = MR.sorted_registry_names()
    groups = _partition_mazes(pool, 4)

    assert len(groups) == 4
    assert sorted(m for g in groups for m in g) == sorted(pool)
    assert all(groups), "a robot with no maze cannot reset"
    assert max(map(len, groups)) - min(map(len, groups)) <= 1


def test_every_robot_cycles_through_its_whole_maze_group(tmp_path, monkeypatch):
    """_train must hand each env its GROUP, not a single maze.

    Passing maze_names=[one_maze] pins the robot to that maze for the whole
    run, which is how 9 of the 13 mazes went untouched while the fleet was
    capped at 4 robots.
    """
    from rl_training import train_explore_multi as trainer
    model = SAC('MlpPolicy', TinyExploreEnv(), learning_starts=0,
                buffer_size=100, batch_size=4,
                policy_kwargs={'net_arch': [8]}, device='cpu')
    model.learn(4)
    checkpoint = tmp_path / 'seed.zip'
    model.save(checkpoint)

    calls = []

    def spy(**kw):
        calls.append(kw)
        return TinyExploreEnv()

    monkeypatch.setattr(trainer, '_make_env', spy)
    args = Namespace(run_dir=tmp_path / 'run', visit_memory=False, seed=0,
                     stall_limit=None, roam_bonus=None, n_robots=2, bootstrap_scripted=0,
                     load=str(checkpoint), load_buffer=None, gradient_steps=1,
                     ckpt_every=100, reset_elites=False, steps=8)
    trainer._train(args, ['ortho_1', 'ortho_2', 'ortho_3'])

    assert [kw['robot_id'] for kw in calls] == [1, 2]
    assert [kw['mazes'] for kw in calls] == [['ortho_1', 'ortho_3'],
                                             ['ortho_2']]
