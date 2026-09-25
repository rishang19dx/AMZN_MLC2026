#!/usr/bin/env bash
# Pack the raw train/test TSVs into ONE zstd archive for upload to Google Drive.
#
# Why one archive: Drive (and the Colab Drive mount) is slow per file and slow
# for streaming reads, so we upload a single compressed file, copy it to the
# Colab VM's local disk and extract there. zstd -19 gets ~3.5x on this data
# (~2.5 GB -> ~0.75 GB); the random entity IDs limit how far it compresses.
#
# The local_train/local_val splits are NOT packed: data_loader.py regenerates
# them deterministically on the VM in about a minute. Their checksums are packed
# (SPLITS.sha256) so cloud_setup.sh can prove the regenerated splits are
# byte-identical to the local ones.
#
# Usage:  bash scripts/pack_data.sh [output_path]      (LEVEL=10 for a faster, larger archive)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA="${BER_DATA_DIR:-$REPO/dataset}"
OUT="${1:-$REPO/mlc26_data.tar.zst}"
LEVEL="${LEVEL:-19}"

cd "$DATA"
echo "Checksumming raw files in $DATA ..."
sha256sum train/*.tsv test/*.tsv > MANIFEST.sha256
extra=()
if compgen -G "splits/*/*.tsv" > /dev/null; then
    sha256sum splits/*/*.tsv > SPLITS.sha256
    extra+=(SPLITS.sha256)
fi

echo "Compressing (zstd -$LEVEL) -> $OUT ..."
tar -cf - MANIFEST.sha256 "${extra[@]}" train/*.tsv test/*.tsv | zstd -q -"$LEVEL" -T0 -o "$OUT" -f
( cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256" )

ls -lh "$OUT"
echo "Upload $(basename "$OUT") and $(basename "$OUT").sha256 to the same Drive folder."
