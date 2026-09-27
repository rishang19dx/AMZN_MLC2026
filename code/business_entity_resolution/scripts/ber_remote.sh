#!/usr/bin/env bash
# Runs ON AIRAWAT. Lean, quota-safe setup for CPU jobs (blocking, experiments).
#
# Rules this script enforces:
#   * Everything lives in $BER_ROOT (default /home/s25017/scratch/s25017), in the
#     sub-folders we create there: repo/ venv/ jobs/ tmp/ .cache/ (marker file
#     .ber_owned). Other files in $BER_ROOT are never written or deleted; `clean`
#     only removes jobs/<split>, which we created.
#   * Home quota: soft limit 9.2 GB. Nothing starts unless it fits under the
#     soft limit with a margin, and a watchdog stops a running job before the
#     soft limit is reached.
#   * Shared node: 48 pinned CPUs (taskset), thread caps, nice 10. No GPU.
#
#   bash ber_remote.sh setup                        # lean CPU venv (~0.45 GB) + tests
#   bash ber_remote.sh headroom                     # MB we may still use (soft limit - margin)
#   bash ber_remote.sh run <split> <name> <cmd...>  # background job with quota watchdog
#   bash ber_remote.sh status                       # quota, our folder sizes, running jobs
#   bash ber_remote.sh clean <split>                # delete OUR job folder for <split> (after pulling)
set -euo pipefail

BER_ROOT="${BER_ROOT:-/home/s25017/scratch/s25017}"
MARGIN_MB="${BER_MARGIN_MB:-300}"      # never plan to come closer than this to the soft limit
GUARD_MB="${BER_GUARD_MB:-150}"        # watchdog stops a job this close to the soft limit
CPUS="${BER_CPUS:-0-47}"
NT="${BER_NTHREADS:-48}"
REPO="$BER_ROOT/repo/code/business_entity_resolution"

die() { echo "ERROR: $*" >&2; exit 1; }
owned() { [ -f "$BER_ROOT/.ber_owned" ] || die "no .ber_owned marker in $BER_ROOT - run 'airawat.sh <target> code' from the laptop first"; }

# used and soft-limit KB of the quota that holds $BER_ROOT (home filesystem)
quota_kb() {
    quota -w 2> /dev/null | awk '$1 ~ /^\// { gsub(/\*/, "", $2); print $2, $3; exit }'
}
headroom_mb() {
    read -r used soft < <(quota_kb) || die "cannot read quota"
    [ -n "${soft:-}" ] && [ "$soft" -gt 0 ] || die "no soft quota found in: $(quota -w 2>&1 | tail -1)"
    echo $(( (soft - used) / 1024 - MARGIN_MB ))
}

job_env() {            # data/cache/output of one split's job folder; caches + temp inside $BER_ROOT
    local split="$1" j="$BER_ROOT/jobs/$1"
    export BER_DATA_DIR="$j/dataset" BER_CACHE_DIR="$j/cache" BER_OUTPUT_DIR="$j/output"
    export TMPDIR="$BER_ROOT/tmp" XDG_CACHE_HOME="$BER_ROOT/.cache"
    export BER_WORKERS="$NT" OMP_NUM_THREADS="$NT" OPENBLAS_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT"
    export NUMEXPR_MAX_THREADS="$NT" BER_BLOCK_MEM_GB="${BER_BLOCK_MEM_GB:-48}" BER_DUCKDB_MEM="${BER_DUCKDB_MEM:-64GB}"
    export CUDA_VISIBLE_DEVICES=""       # CPU jobs only
    mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$BER_OUTPUT_DIR" "$j/logs"
}

cmd="${1:-}"; shift || true
case "$cmd" in
setup)
    owned
    [ "$(headroom_mb)" -gt 700 ] || die "only $(headroom_mb) MB below the soft limit; setup needs ~700 MB"
    mkdir -p "$BER_ROOT/tmp" "$BER_ROOT/.cache"
    export TMPDIR="$BER_ROOT/tmp" XDG_CACHE_HOME="$BER_ROOT/.cache" PIP_NO_CACHE_DIR=1
    if [ ! -x "$BER_ROOT/venv/bin/python" ]; then python3 -m venv "$BER_ROOT/venv"; fi
    # CPU-only subset of requirements.txt (no torch/transformers: GPU work stays off this machine)
    grep -E '^(numpy|pandas|scipy|scikit-learn|duckdb|anyascii|tqdm|RapidFuzz|lightgbm)==' "$REPO/requirements.txt" \
        | "$BER_ROOT/venv/bin/pip" install -q --no-cache-dir -r /dev/stdin
    rm -rf "$BER_ROOT/tmp"/* 2> /dev/null || true
    cd "$REPO"
    "$BER_ROOT/venv/bin/python" tests/test_evaluate.py | tail -1
    "$BER_ROOT/venv/bin/python" tests/test_match.py | tail -1
    echo "venv $(du -sh "$BER_ROOT/venv" | cut -f1); headroom now $(headroom_mb) MB below soft limit - margin"
    ;;
headroom)
    headroom_mb ;;
run)
    owned
    split="${1:?usage: run <split> <name> <cmd...>}"; name="${2:?name}"; shift 2
    [ $# -gt 0 ] || die "no command"
    [ -d "$BER_ROOT/jobs/$split/dataset" ] || die "no data for $split (push it from the laptop first)"
    [ -x "$BER_ROOT/venv/bin/python" ] || die "run setup first"
    job_env "$split"
    log="$BER_ROOT/jobs/$split/logs/${name}_$(date +%m%d_%H%M).log"
    echo "# $(date '+%F %T') | cpus $CPUS | threads $NT | headroom $(headroom_mb) MB | $*" > "$log"
    cd "$REPO"
    export PATH="$BER_ROOT/venv/bin:$PATH"
    timer=(); [ -x /usr/bin/time ] && timer=(/usr/bin/time -f "TOTAL wall %es peakRSS %MKB")
    setsid nohup taskset -c "$CPUS" nice -n 10 "${timer[@]}" "$@" >> "$log" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "${log%.log}.pid"
    # quota watchdog: stop the whole job (its process group) before the soft limit
    setsid nohup bash -c '
        pid=$1; log=$2; guard=$3
        while kill -0 "$pid" 2> /dev/null; do
            read -r used soft < <(quota -w 2> /dev/null | awk '"'"'$1 ~ /^\// { gsub(/\*/, "", $2); print $2, $3; exit }'"'"')
            if [ -n "$soft" ] && [ $(( (soft - used) / 1024 )) -lt "$guard" ]; then
                echo "!! QUOTA WATCHDOG: $(( (soft - used) / 1024 )) MB left below the soft limit - stopping the job" >> "$log"
                kill -TERM -- "-$pid" 2> /dev/null; sleep 20; kill -KILL -- "-$pid" 2> /dev/null; exit
            fi
            sleep 10
        done' _ "$pid" "$log" "$GUARD_MB" > /dev/null 2>&1 < /dev/null &
    echo "started pid $pid (watchdog on) -> $log"
    echo "follow: tail -f $log      stop: kill -- -$pid"
    ;;
status)
    echo "quota: $(quota -s 2> /dev/null | tail -1)"
    [ -d "$BER_ROOT" ] && du -sh "$BER_ROOT"/venv "$BER_ROOT"/jobs/* 2> /dev/null || echo "(no $BER_ROOT yet)"
    echo "headroom: $(headroom_mb) MB"
    for f in "$BER_ROOT"/jobs/*/logs/*.pid; do
        [ -e "$f" ] || continue
        p=$(cat "$f"); if kill -0 "$p" 2> /dev/null; then s=RUNNING; else s=finished; fi
        echo "  $s  pid $p  ${f%.pid}.log"; tail -2 "${f%.pid}.log" | sed 's/^/      /'
    done
    ;;
clean)
    owned
    split="${1:?usage: clean <split>}"; j="$BER_ROOT/jobs/$split"
    [ -d "$j" ] || die "no job folder $j"
    for f in "$j"/logs/*.pid; do [ -e "$f" ] && kill -0 "$(cat "$f")" 2> /dev/null && die "a job is still running in $j"; done
    du -sh "$j"; rm -rf -- "$j"; echo "removed $j (ours); headroom now $(headroom_mb) MB"
    ;;
*)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
