#!/usr/bin/env bash
# Runs on the LAPTOP. Moves code and one job's data to airawat and results back,
# within the 10 GB home quota (soft limit 9.2 GB) - see scripts/ber_remote.sh.
#
#   bash scripts/airawat.sh <ssh-target> code          # first: push code into /home/s25017/scratch/s25017, set up venv
#   bash scripts/airawat.sh <ssh-target> data <split>  # push one split's inputs (gzipped) if they fit
#   bash scripts/airawat.sh <ssh-target> pull <split>  # copy results back to <repo>/from_airawat/<split>/
#   bash scripts/airawat.sh <ssh-target> status
#
# <ssh-target>: what you type after `ssh`, e.g. s25017@airawat. One password
# prompt per call (the connection is reused for ~15 min).
# Jobs are started ON AIRAWAT with ber_remote.sh run ... (printed after `data`).
set -euo pipefail

TARGET="${1:?usage: airawat.sh <ssh-target> code|data <split>|pull <split>|status}"
CMD="${2:?command}"
ROOT="${BER_REMOTE_ROOT:-/home/s25017/scratch/s25017}"   # our sub-folders go in here: repo venv jobs tmp .cache
REMOTE="$ROOT/repo/code/business_entity_resolution/scripts/ber_remote.sh"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJ="$REPO/code/business_entity_resolution"
DATA="$REPO/dataset"

mkdir -p "$HOME/.ssh"
SSH=(ssh -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-ber-%r@%h:%p" -o ControlPersist=15m)
rs() { rsync -a --info=progress2 --human-readable -e "${SSH[*]}" "$@"; }
remote() { "${SSH[@]}" "$TARGET" "$@"; }

# output space each job needs on airawat, beyond its inputs (MB; laptop sizes scaled to K=30/50):
#   test = blocking only; scale_val = blocking + features + cand_index + --cv/--fit outputs
reserve_mb() { case "$1" in test) echo 2400 ;; scale_val) echo 2200 ;; local_train) echo 300 ;; *) echo 1000 ;; esac; }

case "$CMD" in
code)
    # first time: refuse if any name we would create already exists (never adopt someone's files)
    remote "if [ ! -f $ROOT/.ber_owned ]; then
                for d in repo venv jobs tmp .cache; do [ -e $ROOT/\$d ] && { echo \"ERROR: $ROOT/\$d already exists and is not ours\"; exit 1; }; done
            fi
            mkdir -p $ROOT/repo/code && touch $ROOT/.ber_owned"
    rs --exclude='.venv' --exclude='__pycache__' --exclude='notebooks' \
       "$PROJ" "$TARGET:$ROOT/repo/code/"
    rs "$REPO/utils" "$TARGET:$ROOT/repo/"
    echo "code pushed; setting up the venv on airawat (~2-3 min) ..."
    remote "bash $REMOTE setup"
    ;;
data)
    split="${3:?usage: data <split>}"
    case "$split" in test) src="$DATA/test"; rel="test" ;; *) src="$DATA/splits/$split"; rel="splits/$split" ;; esac
    [ -d "$src" ] || { echo "no $src"; exit 1; }
    stage="$REPO/transfer/$split"
    mkdir -p "$stage/dataset/$rel" "$stage/cache"
    cp "$REPO/cache/translit.json" "$stage/cache/"
    for f in "$src"/*_source{1,2,3}.tsv; do                    # sources gzipped (symlinks followed)
        out="$stage/dataset/$rel/$(basename "$f").gz"
        [ -s "$out" ] && [ "$out" -nt "$(readlink -f "$f")" ] || { echo "gzip $(basename "$f") ..."; gzip -c "$f" > "$out.tmp" && mv "$out.tmp" "$out"; }
    done
    [ -e "$src/${split}_ground_truth.tsv" ] && cp -L "$src/${split}_ground_truth.tsv" "$stage/dataset/$rel/"
    need=$(( $(du -sm "$stage" | cut -f1) + $(reserve_mb "$split") ))
    room=$(remote "bash $REMOTE headroom")
    echo "$split: inputs $(du -sh "$stage" | cut -f1) + $(reserve_mb "$split") MB for outputs = $need MB; room on airawat: $room MB"
    [ "$need" -le "$room" ] || { echo "does not fit under the soft limit - clean a finished job first (ber_remote.sh clean <split>)"; exit 1; }
    remote "[ -f $ROOT/.ber_owned ] && mkdir -p $ROOT/jobs/$split" || { echo "run 'code' first"; exit 1; }
    rs "$stage/" "$TARGET:$ROOT/jobs/$split/"
    echo "pushed. On airawat, e.g.:"
    echo "  bash $REMOTE run $split <name> python src/blocking.py --split $split --no-tsv"
    ;;
pull)
    split="${3:?usage: pull <split>}"
    dest="$REPO/from_airawat/$split"; mkdir -p "$dest"
    rs --exclude='duckdb_tmp' --exclude='blocking_parts' "$TARGET:$ROOT/jobs/$split/cache" "$TARGET:$ROOT/jobs/$split/logs" "$dest/"
    rs "$TARGET:$ROOT/jobs/$split/output" "$dest/" 2> /dev/null || true
    echo "results in $dest (the laptop's own cache/ is not touched):"; find "$dest" -type f | sed "s|$REPO/||"
    ;;
status)
    remote "bash $REMOTE status" ;;
*)
    sed -n '2,14p' "$0"; exit 1 ;;
esac
