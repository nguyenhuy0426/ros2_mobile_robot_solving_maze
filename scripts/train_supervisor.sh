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
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$WS_ROOT"
source /opt/ros/jazzy/setup.bash
[[ -f install/setup.bash ]] && source install/setup.bash
PYTHON=${PYTHON:-$WS_ROOT/.venv/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3

ROOT=${1:?usage: train_supervisor.sh <campaign-root> [args...]}
shift
mkdir -p "$ROOT"
LOG="$ROOT/supervisor.log"
# A 2-day campaign appends non-stop; rotate instead of growing without bound.
if [[ -f "$LOG" && $(stat -c%s "$LOG") -gt 20971520 ]]; then
    mv -f "$LOG" "$LOG.old"
fi
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

# Resume-target selection. `ls -t *.zip` picks sac_explore_multi_final.zip,
# because the trainer writes it LAST (in its own finally block) -- so every
# restart warm-started from the most recent, often already-degenerated policy
# and the elite checkpoints were never used. HANDOFF 3.4 is explicit that
# final.zip must never be a seed. Instead rank every attempt's elites.json by
# the same key the callback uses, (coverage_frac, score), and take the best
# explorer actually recorded.
best_ckpt() {
    "$PYTHON" - "$ROOT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
best = None  # (coverage, score, mtime, path)
for js in glob.glob(os.path.join(root, "attempt_*", "logs", "elites.json")):
    try:
        data = json.load(open(js))
    except Exception:
        continue
    adir = os.path.dirname(os.path.dirname(js))
    for entry in data.get("mazes", {}).values():
        path = os.path.join(adir, "checkpoints", str(entry.get("file", "")))
        if not os.path.isfile(path):
            continue
        key = (float(entry.get("coverage_frac", 0.0)),
               float(entry.get("score", float("-inf"))),
               os.path.getmtime(path))
        if best is None or key > best[0]:
            best = (key, path)
if best:
    print(best[1])
PY
}

while true; do
    RUN="$ROOT/attempt_$(date +%Y%m%d_%H%M%S)"
    # On a resume the buffer already holds real transitions, so re-running
    # --bootstrap-scripted dumps another load of the demonstrator's off-policy
    # rollouts on top (measured: ~75% scripted by the 5th attempt), diluting
    # the on-policy data SAC needs. Seed the cold start only.
    TRAIN_ARGS=("$@")
    if [[ -n "$BUF" ]]; then
        filtered=()
        skip=0
        for a in "$@"; do
            if [[ $skip -eq 1 ]]; then skip=0; continue; fi
            if [[ "$a" == "--bootstrap-scripted" ]]; then skip=1; continue; fi
            filtered+=("$a")
        done
        TRAIN_ARGS=("${filtered[@]}")
    fi
    echo "=== $(date -Is) starting trainer in $RUN ${CKPT:+(resume $CKPT)}" \
         "${BUF:+(buffer $BUF, bootstrap dropped)} ===" >> "$LOG"
    "$PYTHON" -m rl_training.train_explore_multi \
        --run-dir "$RUN" ${CKPT:+--load "$CKPT"} \
        ${BUF:+--load-buffer "$BUF"} "${TRAIN_ARGS[@]}" >> "$LOG" 2>&1
    rc=$?
    echo "=== $(date -Is) trainer exited rc=$rc ===" >> "$LOG"
    [[ $rc -eq 0 ]] && break
    # Prefer the best elite across attempts; fall back to newest non-final
    # checkpoint only if no elites.json exists anywhere yet.
    BEST=$(best_ckpt)
    if [[ -n "$BEST" ]]; then
        CKPT="$BEST"
        # Keep the replay buffer paired with the elite's own policy. A newer
        # buffer may belong to a later, already-different policy; feeding it
        # to this checkpoint makes the resumed critic bootstrap from an
        # inconsistent transition distribution.
        SAME_BUF="$(dirname "$BEST")/replay_buffer.pkl"
        if [[ -f "$SAME_BUF" ]]; then
            BUF="$SAME_BUF"
        fi
    else
        NEW=$(ls -t "$ROOT"/attempt_*/checkpoints/*.zip 2>/dev/null | head -1)
        [[ -n "$NEW" ]] && CKPT="$NEW"
        # With no elite table yet, use the newest checkpoint and its paired
        # buffer from the same attempt when available.
        if [[ -n "$NEW" ]]; then
            NEWBUF="$(dirname "$NEW")/replay_buffer.pkl"
            [[ -f "$NEWBUF" ]] && BUF="$NEWBUF"
        fi
    fi
    # Disk guard: every attempt owns checkpoints (~100 model zips) + a replay
    # buffer pickle, so a crash-looping campaign grows ~5 MB/attempt. Keep the
    # newest KEEP_ATTEMPTS; the resume selection above already read the best
    # elite and newest buffer, and they live in the newest dirs anyway.
    KEEP_ATTEMPTS=12
    ls -1dt "$ROOT"/attempt_* 2>/dev/null | tail -n +$((KEEP_ATTEMPTS + 1)) \
        | while read -r old; do
            echo "=== $(date -Is) pruning old attempt $old" >> "$LOG"
            rm -rf "$old"
        done
    sleep 20
done
