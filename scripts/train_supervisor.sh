#!/usr/bin/env bash
# Supervisor for a multi-robot explore training campaign.
#
# train_explore_multi refuses to reuse a run directory on purpose (every run
# owns an isolated hall of fame and CSV), so a restart cannot simply re-enter
# the old one. Instead each attempt gets its own timestamped run dir under
# $ROOT and warm-starts from the previous attempt's newest checkpoint AND its
# replay buffer, so a crash costs neither the policy nor the data.
#
# Carrying the buffer is not an optimisation, it is what makes the loop learn
# at all: a .zip-only warm start pushes learning_starts to
# num_timesteps + SACW_LEARN_STARTS, so an attempt that dies before that
# threshold does ZERO gradient updates and acts uniformly at random for its
# whole life. A campaign of short attempts then burns hours and learns
# nothing, which is exactly what the 13-robot campaign did.
#
# Gazebo itself is kept alive by scripts/train_watchdog.sh — that is the half
# this loop cannot repair, since every robot and bridge dies with the server.
#
# Usage: ./scripts/train_supervisor.sh <campaign-root> [extra train args...]
set +u
cd /home/huynn/ros2_gazebo/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ROOT=${1:?usage: train_supervisor.sh <campaign-root> [args...]}
shift
mkdir -p "$ROOT"
LOG="$ROOT/supervisor.log"
# SEED_CKPT warm-starts the FIRST attempt from a checkpoint outside $ROOT
# (e.g. the previous campaign's elite). It is deliberately not a positional
# arg: those are forwarded verbatim to every attempt, so a --load passed
# that way would override the resume checkpoint on every restart and pin the
# campaign to its starting policy forever.
#
# SEED_BUF pairs a replay buffer with it, and defaults to EMPTY because a seed
# from an earlier campaign carries transitions from the OLD dynamics and the
# OLD reward, which would poison the critic rather than warm it. Set it only
# when the seed came from a run whose reward function is identical to this
# one's — the case it exists for is cycling a LIVE campaign onto new control
# code, where dropping ~68k on-policy transitions costs far more than the
# small dynamics shift the new code introduces. A graceful shutdown returns
# rc=0 and so ends this loop by design (that is what a deliberate Ctrl-C
# means), which is why re-entering a campaign has to come back through here.
CKPT="${SEED_CKPT:-}"
BUF="${SEED_BUF:-}"

while true; do
    RUN="$ROOT/attempt_$(date +%Y%m%d_%H%M%S)"
    echo "=== $(date -Is) starting trainer in $RUN ${CKPT:+(resume $CKPT)}" \
         "${BUF:+(buffer $BUF)} ===" >> "$LOG"
    rl_venv/bin/python -m rl_training.train_explore_multi \
        --run-dir "$RUN" ${CKPT:+--load "$CKPT"} \
        ${BUF:+--load-buffer "$BUF"} "$@" >> "$LOG" 2>&1
    rc=$?
    echo "=== $(date -Is) trainer exited rc=$rc ===" >> "$LOG"
    [[ $rc -eq 0 ]] && break
    NEW=$(ls -t "$ROOT"/attempt_*/checkpoints/*.zip 2>/dev/null | head -1)
    [[ -n "$NEW" ]] && CKPT="$NEW"
    # Pair the buffer with the checkpoint: an attempt that died before
    # writing one leaves the previous buffer in place, which is stale
    # relative to CKPT but still better than restarting the warmup.
    NEWBUF=$(ls -t "$ROOT"/attempt_*/checkpoints/replay_buffer.pkl \
             2>/dev/null | head -1)
    [[ -n "$NEWBUF" ]] && BUF="$NEWBUF"
    sleep 20
done
