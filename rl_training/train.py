#!/usr/bin/env python3
"""
train.py — DQN training for 5×5 maze solver with lidar input.

Episode-based training loop adapted from DRL-Robot-Navigation-ROS2.
Uses replay buffer for off-policy learning with Double DQN.

By default NO classical solver participates in training (pure NN + RL):
random warmup, no behavior cloning, no expert-episode injection. Pass
--use-expert to re-enable the BFS expert warm-start machinery.

Usage:
  # Pure-RL training in the fast simulator:
  rl_venv/bin/python3 -m rl_training.train

  # Training in real Gazebo:
  rl_venv/bin/python3 -m rl_training.train --gazebo

  # Legacy mode with BFS-expert warmup / BC / injection (not pure RL):
  rl_venv/bin/python3 -m rl_training.train --use-expert
"""

import argparse
import os
import time
import numpy as np
from pathlib import Path

from rl_training.maze_env import MazeEnv, CurriculumMazeEnv
from rl_training.replay_buffer import ReplayBuffer
from rl_training.elite_buffer import EliteBuffer
from rl_training.dqn import DQN
from rl_training import config as C
from rl_training.expert import get_expert_action


class MixedBufferWrapper:
    """Wrapper to mix samples from regular and elite buffers."""
    def __init__(self, regular_buffer, elite_buffer, elite_ratio=0.2):
        self.reg = regular_buffer
        self.elite = elite_buffer
        self.elite_ratio = elite_ratio

    def sample(self, batch_size):
        n_elite = int(batch_size * self.elite_ratio)
        n_reg = batch_size - n_elite

        reg_batch = self.reg.sample(n_reg)
        elite_batch = self.elite.sample(n_elite)

        if elite_batch is None:
            return self.reg.sample(batch_size)

        states = np.concatenate((reg_batch[0], elite_batch[0]), axis=0)
        actions = np.concatenate((reg_batch[1], elite_batch[1]), axis=0)
        rewards = np.concatenate((reg_batch[2], elite_batch[2]), axis=0)
        dones = np.concatenate((reg_batch[3], elite_batch[3]), axis=0)
        next_states = np.concatenate((reg_batch[4], elite_batch[4]), axis=0)

        return states, actions, rewards, dones, next_states


# ═══════════════════════════════════════════════════════════════════════════
# Behavior-cloning warm-start
# ═══════════════════════════════════════════════════════════════════════════

def bc_pretrain(model,
                n_episodes: int = C.BC_WARMSTART_EPISODES,
                epochs: int = C.BC_WARMSTART_EPOCHS):
    """Supervised pretraining of the Q-net on BFS-expert demonstrations.

    Maps frame-stacked observations → expert action (cross-entropy over Q-logits),
    so RL starts from a competent navigator instead of first having to discover the
    passive spin-in-place optimum. Trains model.q_net in place and syncs the target.
    """
    import torch
    import torch.nn as nn

    print(f"\n  Behavior-cloning warm-start: collecting {n_episodes} expert episodes...")
    X, Y = [], []
    for i in range(n_episodes):
        env = MazeEnv(curriculum_level=i % 3)  # mix easy/medium/hard layouts
        obs, info = env.reset(seed=i + 1000)
        for _ in range(C.MAX_STEPS):
            a = get_expert_action(env)
            X.append(obs.copy())
            Y.append(a)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int64)
    dev = model.device
    Xt = torch.as_tensor(X, device=dev)
    Yt = torch.as_tensor(Y, device=dev)
    opt = torch.optim.Adam(model.q_net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    n = len(Xt)
    bs = 256

    print(f"  collected {n:,} (obs, action) pairs; training {epochs} supervised epochs...")
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for j in range(0, n, bs):
            idx = perm[j:j + bs]
            loss = lossf(model.q_net(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        acc = (model.q_net(Xt).argmax(1) == Yt).float().mean().item()
    model.q_target.load_state_dict(model.q_net.state_dict())
    print(f"  BC warm-start done: train action-match acc={acc:.1%}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train(max_episodes: int = 5000,
          load_path: str = None,
          curriculum: bool = True,
          warmup_mode: str = "random",
          bc_warmstart: bool = False,
          gazebo: bool = False,
          use_expert: bool = False):
    """
    DQN training loop on random mazes (or real Gazebo).

    Follows the DRL-Robot-Navigation-ROS2 episode-based pattern:
      1. Run episode → collect experiences
      2. Every N episodes → train from replay buffer
      3. Every M episodes → evaluate and save

    ``use_expert`` re-enables the legacy BFS-expert machinery (expert warmup,
    BC pretraining, expert-episode injection). Default is pure RL: the policy
    learns only from its own experience, per the project's no-solver rule.
    """
    if use_expert:
        warmup_mode = "expert"
        print("  ⚠ BFS-expert assistance ENABLED (--use-expert): warmup, "
              "BC pretraining and expert-episode injection are active.")

    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    C.TB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  DQN Mecanum Maze Solver Training                   ║")
    print(f"  ║  Episodes: {max_episodes:<6d}  Max steps/ep: {C.MAX_STEPS}          ║")
    print(f"  ║  Obs: {C.STATE_DIM} dim ({C.FRAME_STACK}×{C.SINGLE_OBS_DIM} stack + {C.N_ACTIONS} last-act)    ║")
    print(f"  ║  Act: {C.N_ACTIONS} discrete (cardinal + turn)              ║")
    print(f"  ║  Net: {C.STATE_DIM}→{C.HIDDEN_1}→{C.HIDDEN_2}→{C.ACTION_DIM} (Dueling DQN)        ║")
    print(f"  ║  Replay: {C.BUFFER_SIZE:,} | Batch: {C.BATCH_SIZE}                   ║")
    print(f"  ║  Environment: {'GAZEBO' if gazebo else ('CURRICULUM' if curriculum else 'RANDOM')}                ║")
    print(f"  ╚══════════════════════════════════════════════════════╝\n")

    # Model
    model = DQN(load_model=bool(load_path),
                load_path=Path(load_path) if load_path else None)

    print(f"  Algorithm: Double DQN (Dueling)")
    print(f"  Model parameters: {model.total_params:,}")
    print(f"  Device: {model.device}\n")

    # Behavior-cloning warm-start (fresh fast-sim runs only — skip when resuming a
    # checkpoint or training in Gazebo). Leaves the policy as a competent navigator
    # and lowers the starting epsilon so RL exploits it instead of re-exploring.
    if bc_warmstart and not gazebo and not load_path:
        bc_pretrain(model)
        model.epsilon = C.BC_EPS_START
        print(f"  Epsilon set to {model.epsilon} (exploit the warm-started policy).\n")

    # Replay buffers
    replay_buffer = ReplayBuffer(buffer_size=C.BUFFER_SIZE)
    elite_buffer = EliteBuffer(buffer_size=C.ELITE_BUFFER_SIZE)

    # Environment
    if gazebo:
        from rl_training.gazebo_env import GazeboMazeEnv
        env = GazeboMazeEnv()
        curriculum = False
        warmup_mode = "random"
    elif curriculum:
        env = CurriculumMazeEnv()
    else:
        env = MazeEnv()

    # Tracking
    total_steps = 0
    train_count = 0
    best_eval_reward = -np.inf
    best_eval_success = -np.inf
    best_success_avg = 0.0
    degraded_count = 0
    post_warmup_eps = 0  # model-driven episodes since warmup (excludes expert episodes)
    ROLLBACK_RATIO = 0.60
    PATIENCE_STEPS = 15
    ep_rewards = []
    ep_successes = []
    ep_collisions = []
    ep_cells = []
    expert_rng = np.random.default_rng(12345)  # deterministic expert-episode coin flips
    t0 = time.time()
    episode = 0

    try:
        for episode in range(1, max_episodes + 1):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            ep_transitions = []

            # Decaying expert-episode injection (only with --use-expert): run a
            # fraction of whole episodes with the BFS expert so optimal
            # goal-reaching trajectories keep flowing into the buffers while
            # the model is still mostly exploring.
            started_post_warmup = total_steps >= C.WARMUP_STEPS
            expert_frac = C.EXPERT_FRAC_START * max(0.0, 1.0 - episode / C.EXPERT_DECAY_EPISODES)
            is_expert_episode = (use_expert and started_post_warmup and not gazebo
                                 and expert_rng.random() < expert_frac)
            is_model_episode = started_post_warmup and not is_expert_episode

            for step in range(C.MAX_STEPS):
                total_steps += 1

                # Select action
                if total_steps < C.WARMUP_STEPS:
                    if warmup_mode == "expert" and not gazebo:
                        action = get_expert_action(env)
                    else:
                        action = env.action_space.sample()
                else:
                    if total_steps == C.WARMUP_STEPS:
                        print("\n  🚀 Warmup complete! Resetting curriculum to Level 0 for model training.")
                        if hasattr(env, "_current_level"):
                            env._current_level = 0
                            env._success_history.clear()
                    if is_expert_episode:
                        action = get_expert_action(env)
                    else:
                        action = model.get_action(obs, add_noise=True)

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated  # for replay buffer (truncated = not terminal)

                # Store experience (action is an integer)
                replay_buffer.add(obs, action, reward, done, next_obs)
                ep_transitions.append((obs, action, reward, done, next_obs))

                obs = next_obs
                episode_reward += reward
                episode_steps += 1

                if terminated or truncated:
                    break

            # Decay epsilon after each post-warmup episode
            if started_post_warmup:
                model.decay_epsilon()

            # Store SUCCESSFUL episodes (expert OR model) in the elite buffer —
            # these are the optimal goal-reaching trajectories we want to
            # over-sample.
            cells_visited = info.get("cells_visited", 0)
            success = info.get("success", False)
            if elite_buffer.should_store(success):
                elite_buffer.store_episode(ep_transitions)

            # Track results — MODEL episodes only, so expert-driven episodes
            # don't inflate the model's measured success/reward.
            if is_model_episode:
                ep_rewards.append(episode_reward)
                ep_successes.append(success)
                ep_collisions.append(bool(info.get("collision", False)
                                          or info.get("collision_bounce", False)))
                ep_cells.append(cells_visited)

            # Train every N episodes
            if (episode % C.TRAIN_EVERY_N_EP == 0 and
                    len(replay_buffer) >= C.BATCH_SIZE):

                # Check if we should mix with elite buffer
                if len(elite_buffer) >= C.BATCH_SIZE:
                    mixed_buffer = MixedBufferWrapper(replay_buffer, elite_buffer, C.ELITE_SAMPLE_RATIO)
                    model.train(mixed_buffer,
                                iterations=C.TRAIN_ITERATIONS,
                                batch_size=C.BATCH_SIZE)
                else:
                    model.train(replay_buffer,
                                iterations=C.TRAIN_ITERATIONS,
                                batch_size=C.BATCH_SIZE)
                train_count += 1

            # Save best-success checkpoint (GA rollback target) on a moving average.
            # Only assess once the window is fully post-warmup, so expert warmup
            # episodes don't inflate the baseline the (untrained) model is judged against.
            if is_model_episode:
                post_warmup_eps += 1
            if is_model_episode and post_warmup_eps >= C.ELITE_AVG_WINDOW:
                avg_success = np.mean(ep_successes[-C.ELITE_AVG_WINDOW:])

                if avg_success > best_success_avg:
                    best_success_avg = avg_success
                    model.save(C.CKPT_DIR / "best_explorer")
                    print(f"    ★ New success record: {avg_success:.0%} avg (saved best_explorer)")

                # Genetic Elitism Rollback with Patience — gated on SUCCESS so it stays
                # dormant until the policy actually reaches goals, then guards against
                # catastrophic regression instead of chasing exploration.
                if best_success_avg >= 0.3:  # only once the policy is reliably solving
                    if avg_success < best_success_avg * ROLLBACK_RATIO:
                        degraded_count += 1
                        if degraded_count >= PATIENCE_STEPS:
                            print(f"    ⚠️ Success degraded ({avg_success:.0%} < best {best_success_avg:.0%}). GA Elitism Rollback triggered!")
                            saved_eps = model.epsilon  # preserve current epsilon
                            model.load(C.CKPT_DIR / "best_explorer")
                            model.epsilon = saved_eps  # restore epsilon (don't reset exploration)
                            if hasattr(model, 'perturb_weights'):
                                model.perturb_weights(noise_scale=0.01)
                            degraded_count = 0
                            # Reset recent history to prevent immediate re-triggering
                            for idx in range(1, min(C.ELITE_AVG_WINDOW + 1, len(ep_successes) + 1)):
                                ep_successes[-idx] = best_success_avg
                    else:
                        degraded_count = 0

            # Log periodically (only once we have model-episode stats to report)
            log_due = (episode % 5 == 0) if gazebo else (episode % 50 == 0)
            if log_due and len(ep_rewards) > 0:
                recent = min(50 if not gazebo else 5, len(ep_rewards))
                avg_r = np.mean(ep_rewards[-recent:])
                sr = sum(ep_successes[-recent:]) / recent
                cr = sum(ep_collisions[-recent:]) / recent
                avg_c = np.mean(ep_cells[-recent:])
                elapsed = time.time() - t0
                fps = total_steps / max(1, elapsed)

                print(f"  ep={episode:5d}  steps={total_steps:>8d}  "
                      f"buf={len(replay_buffer):>6d}  elite={len(elite_buffer):>5d}  "
                      f"avg_r={avg_r:7.1f}  cells={avg_c:4.1f}  success={sr:.1%}  col={cr:.1%}  "
                      f"eps={model.epsilon:.3f}  fps={fps:.1f}  time={elapsed:.0f}s")

                # TensorBoard
                model.writer.add_scalar("episode/reward", avg_r, episode)
                model.writer.add_scalar("episode/cells_visited", avg_c, episode)
                model.writer.add_scalar("episode/success_rate", sr, episode)
                model.writer.add_scalar("episode/collision_rate", cr, episode)
                model.writer.add_scalar("episode/buffer_size",
                                        len(replay_buffer), episode)
                model.writer.add_scalar("episode/elite_buffer_size",
                                        len(elite_buffer), episode)
                model.writer.add_scalar("episode/epsilon",
                                        model.epsilon, episode)

            # Evaluate periodically
            if episode % C.EVAL_EVERY_N_EP == 0:
                if gazebo:
                    recent = min(C.EVAL_EVERY_N_EP, len(ep_rewards))
                    avg_r = np.mean(ep_rewards[-recent:])
                    if avg_r > best_eval_reward:
                        best_eval_reward = avg_r
                        model.save(C.CKPT_DIR / "best")
                        print(f"    ★ New best training reward: {avg_r:.1f} (saved to {C.CKPT_DIR}/best)")
                else:
                    current_level = getattr(env, "_current_level", -1)
                    eval_reward, eval_success = _evaluate(model, n_mazes=C.EVAL_EPISODES, curriculum_level=current_level)
                    model.writer.add_scalar("eval/mean_reward", eval_reward, episode)
                    model.writer.add_scalar("eval/success_rate", eval_success, episode)

                    # Select best on SUCCESS rate (tie-break on reward) — the goal is
                    # reaching the exit, not accumulating exploration reward.
                    if (eval_success, eval_reward) > (best_eval_success, best_eval_reward):
                        best_eval_success = eval_success
                        best_eval_reward = eval_reward
                        model.save(C.CKPT_DIR / "best")
                        print(f"    ★ New best: success={eval_success:.1%} "
                              f"reward={eval_reward:.1f} (saved to {C.CKPT_DIR}/best)")

    except KeyboardInterrupt:
        print("\n  Training interrupted — saving checkpoint...")
    finally:
        if gazebo:
            env.close()

    elapsed = time.time() - t0
    model.save(C.CKPT_DIR / "final")

    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  Training complete!                                  ║")
    print(f"  ║  Episodes: {episode}  Steps: {total_steps:,}             ║")
    print(f"  ║  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)               ║")
    print(f"  ║  Best eval reward: {best_eval_reward:.1f}                       ║")
    print(f"  ║  Saved: {C.CKPT_DIR}/final                          ║")
    print(f"  ╚══════════════════════════════════════════════════════╝\n")


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate(model, n_mazes: int = 20, curriculum_level: int = -1, verbose: bool = False):
    """Quick evaluation on random mazes. Returns (mean_reward, success_rate)."""
    rewards = []
    successes = 0
    for i in range(n_mazes):
        env = MazeEnv(curriculum_level=curriculum_level)
        obs, _ = env.reset(seed=i + 50000)
        ep_r = 0.0
        for _ in range(C.MAX_STEPS):
            action = model.get_action(obs, add_noise=False)
            obs, reward, term, trunc, info = env.step(action)
            ep_r += reward
            if term or trunc:
                break
        rewards.append(ep_r)
        if info.get("success"):
            successes += 1

    mean_r = np.mean(rewards)
    sr = successes / n_mazes
    if verbose or True:
        print(f"    Eval ({n_mazes} mazes): reward={mean_r:.1f}  "
              f"success={sr:.1%}  ({successes}/{n_mazes})")
    return mean_r, sr


def evaluate_full(model_path: str, n_mazes: int = 100, render: bool = False):
    """Full evaluation with detailed per-maze output."""
    model = DQN()
    model.load(Path(model_path))

    successes = 0
    collisions = 0
    timeouts = 0
    total_steps = []
    total_rewards = []

    print(f"\n  Evaluating {model_path} on {n_mazes} random mazes...\n")

    for i in range(n_mazes):
        env = MazeEnv(render_mode="ansi" if render else None)
        obs, info = env.reset(seed=i + 10000)
        episode_reward = 0.0

        for step in range(C.MAX_STEPS):
            action = model.get_action(obs, add_noise=False)
            obs, reward, term, trunc, info = env.step(action)
            episode_reward += reward
            if term or trunc:
                break

        if info.get("success"):
            successes += 1
            marker = "✓"
        elif info.get("collision"):
            collisions += 1
            marker = "✗"
        else:
            timeouts += 1
            marker = "⏱"

        total_steps.append(step + 1)
        total_rewards.append(episode_reward)

        if render or (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{n_mazes}] {marker} "
                  f"steps={step+1:3d}  reward={episode_reward:7.1f}  "
                  f"dist={info.get('dist_to_goal', 0):.2f}m")

    sr = successes / n_mazes
    cr = collisions / n_mazes
    print(f"\n  ═══════════════════════════════════════")
    print(f"  Results on {n_mazes} random 5×5 mazes:")
    print(f"    Success rate:   {sr:.1%} ({successes}/{n_mazes})")
    print(f"    Collision rate: {cr:.1%} ({collisions}/{n_mazes})")
    print(f"    Timeout rate:   {timeouts/n_mazes:.1%} ({timeouts}/{n_mazes})")
    print(f"    Avg steps:      {np.mean(total_steps):.0f}")
    print(f"    Avg reward:     {np.mean(total_rewards):.1f}")
    print(f"  ═══════════════════════════════════════\n")

    return sr


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="DQN Maze Solver Training")
    ap.add_argument("--episodes", type=int, default=5000,
                    help="Total training episodes (default 5000)")
    ap.add_argument("--load", type=str, default=None,
                    help="Load model from checkpoint directory")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="Disable curriculum learning")
    ap.add_argument("--use-expert", action="store_true",
                    help="Enable BFS-expert assistance: expert warmup, BC "
                         "pretraining and expert-episode injection "
                         "(default: off — pure NN + RL)")
    ap.add_argument("--warmup-mode", type=str, default=None,
                    choices=["expert", "random"],
                    help="Warmup exploration mode (default: expert if "
                         "--use-expert else random)")
    ap.add_argument("--eval", type=str, default=None,
                    help="Evaluate a trained model (path to checkpoint dir)")
    ap.add_argument("--eval-mazes", type=int, default=100,
                    help="Number of mazes for evaluation")
    ap.add_argument("--eval-render", action="store_true",
                    help="Render during evaluation")
    ap.add_argument("--gazebo", action="store_true",
                    help="Train directly in ROS2 Gazebo instead of fast simulator")
    args = ap.parse_args()

    warmup_mode = args.warmup_mode
    if warmup_mode is None:
        warmup_mode = "expert" if args.use_expert else "random"

    if args.eval:
        evaluate_full(args.eval, n_mazes=args.eval_mazes, render=args.eval_render)
    else:
        train(
            max_episodes=args.episodes,
            load_path=args.load,
            curriculum=not args.no_curriculum,
            warmup_mode=warmup_mode,
            bc_warmstart=args.use_expert,
            gazebo=args.gazebo,
            use_expert=args.use_expert,
        )


if __name__ == "__main__":
    main()
