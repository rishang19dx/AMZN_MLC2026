#!/usr/bin/env bash
# One-shot cloud VM setup (Colab or Kaggle). Run from the cloned repo:
#   bash code/business_entity_resolution/scripts/cloud_setup.sh <archive>
#   Colab:  <archive> = /content/drive/MyDrive/mlc26/mlc26_data.tar.zst
#   Kaggle: <archive> = /kaggle/input/mlc26-data/mlc26_data.tar.zst
#   or a directory that contains train/train_source1.tsv and test/ somewhere below it, e.g. the
#   official student_resource zip uploaded as a Kaggle Dataset (Kaggle extracts it):
#           <archive> = /kaggle/input/<dataset-slug>
#
# 1. copies the archive from Drive to local disk and verifies its checksum
# 2. extracts into $BER_DATA_DIR (default <repo>/dataset) and verifies every file
# 3. installs pinned requirements, keeping the VM's own CUDA build of torch
# 4. regenerates local_train/local_val and checks they are byte-identical to the local machine's
# 5. runs the scorer tests
set -euo pipefail

ARCHIVE="${1:?usage: cloud_setup.sh <path/to/mlc26_data.tar.zst | raw data directory>}"
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PROJ/../.." && pwd)"
DATA="${BER_DATA_DIR:-$REPO/dataset}"
TMP="${TMPDIR:-/tmp}/$(basename "$ARCHIVE")"

mkdir -p "$DATA"
if [ -d "$ARCHIVE" ]; then
    # raw TSVs (read-only input): link train/ and test/ into $DATA; splits are written next to them
    SRC_TRAIN="$(dirname "$(find -L "$ARCHIVE" -name train_source1.tsv -path '*train/*' | head -1)")"
    SRC_TEST="$(dirname "$(find -L "$ARCHIVE" -name test_source1.tsv -path '*test/*' | head -1)")"
    [ -f "$SRC_TRAIN/train_ground_truth.tsv" ] && [ -f "$SRC_TEST/test_source3.tsv" ] \
        || { echo "no train/ and test/ challenge files under $ARCHIVE"; exit 1; }
    ln -sfn "$SRC_TRAIN" "$DATA/train"
    ln -sfn "$SRC_TEST" "$DATA/test"
    echo "[1-2] using raw data: $DATA/train -> $SRC_TRAIN, $DATA/test -> $SRC_TEST"
elif [ -f "$DATA/MANIFEST.sha256" ] && ( cd "$DATA" && sha256sum --quiet -c MANIFEST.sha256 ) 2>/dev/null; then
    echo "[1-2] data already present and verified in $DATA"
else
    echo "[1] copying archive to local disk ..."
    cp "$ARCHIVE" "$TMP"
    if [ -f "$ARCHIVE.sha256" ]; then
        ( cd "$(dirname "$TMP")" && sha256sum -c "$ARCHIVE.sha256" )
    else
        echo "    (no $ARCHIVE.sha256 next to the archive; skipping archive checksum)"
    fi
    echo "[2] extracting to $DATA ..."
    if command -v zstd > /dev/null || apt-get -qq install -y zstd > /dev/null 2>&1; then
        zstd -dc "$TMP" | tar -xf - -C "$DATA"
    else  # no zstd binary and no apt: fall back to the Python bindings
        pip install -q zstandard
        python -c "import sys, zstandard; zstandard.ZstdDecompressor().copy_stream(open(sys.argv[1], 'rb'), sys.stdout.buffer)" "$TMP" \
            | tar -xf - -C "$DATA"
    fi
    rm -f "$TMP"
    ( cd "$DATA" && sha256sum --quiet -c MANIFEST.sha256 ) && echo "    all raw files verified"
fi

echo "[3] installing requirements (keeping the preinstalled CUDA torch) ..."
grep -vE '^torch==' "$PROJ/requirements.txt" | pip install -q -r /dev/stdin
python -c "import torch; print('    torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "[4] regenerating local splits ..."
( cd "$PROJ" && python src/data_loader.py )
if [ -f "$DATA/SPLITS.sha256" ]; then
    ( cd "$DATA" && sha256sum --quiet -c SPLITS.sha256 ) && echo "    splits identical to the local machine"
fi

echo "[5] scorer tests ..."
( cd "$PROJ" && python tests/test_evaluate.py )
echo "Setup complete."
