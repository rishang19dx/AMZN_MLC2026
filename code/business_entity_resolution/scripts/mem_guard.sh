#!/usr/bin/env bash
# Memory guard for long pipeline runs on a shared laptop (IDE, browser, ...).
#
# Polls MemAvailable. If it stays below MIN_MB for two consecutive checks, it
# stops the pipeline cleanly (SIGTERM, then SIGKILL after a grace period)
# BEFORE the kernel's OOM killer picks a victim, which could be the IDE.
# Finished work survives: every stage writes its outputs to disk as it goes and
# run_pipeline.sh skips/resumes finished stages and parts, so rerunning the
# same command continues where it stopped.
#
# Usage (from code/business_entity_resolution/):
#   bash scripts/mem_guard.sh &          # exits by itself when the pipeline ends
#   MIN_MB=1500 bash scripts/mem_guard.sh &
set -u
MIN_MB="${MIN_MB:-1000}"
INTERVAL="${INTERVAL:-5}"
LOG="${LOG:-../../output/mem_guard.log}"
PIPE_PATTERN="${PIPE_PATTERN:-^(\S*/)?bash scripts/run_pipeline\.sh}"   # the driver itself, not shells that mention it
STAGE_PATTERN="${STAGE_PATTERN:-^\S*python\S* src/(data_loader|translit|blocking|features|match|cross_encoder|error_analysis)\.py}"

mkdir -p "$(dirname "$LOG")"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }

say "guard started: stop pipeline if MemAvailable < ${MIN_MB} MB twice in a row (every ${INTERVAL}s)"
low=0
while pgrep -f "$PIPE_PATTERN|$STAGE_PATTERN" > /dev/null; do
    a=$(avail_mb)
    if [ "$a" -lt "$MIN_MB" ]; then
        low=$((low + 1))
        say "low memory: ${a} MB available (strike $low)"
    else
        low=0
    fi
    if [ "$low" -ge 2 ]; then
        say "STOPPING pipeline to protect the system (MemAvailable ${a} MB)"
        pkill -TERM -f "$PIPE_PATTERN"          # first the driver, so no next stage starts
        pkill -TERM -f "$STAGE_PATTERN"
        sleep 10
        pkill -KILL -f "$PIPE_PATTERN"
        pkill -KILL -f "$STAGE_PATTERN"
        say "stopped. Free memory, then rerun: bash scripts/run_pipeline.sh (finished stages/parts are kept)"
        if [ "${SHUTDOWN:-0}" = 1 ]; then
            sync                                   # flush saved parts to disk
            say "SHUTDOWN=1: powering off in 60 s (cancel: pkill -f mem_guard.sh)"
            sleep 60
            sync
            systemctl poweroff
        fi
        exit 1
    fi
    sleep "$INTERVAL"
done
say "pipeline finished; guard exiting"
if [ "${SHUTDOWN_ON_FINISH:-0}" = 1 ]; then
    sync
    say "SHUTDOWN_ON_FINISH=1: powering off in 60 s (cancel: pkill -f mem_guard.sh)"
    sleep 60
    sync
    systemctl poweroff
fi
