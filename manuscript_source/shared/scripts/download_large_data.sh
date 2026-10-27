#!/usr/bin/env bash
# Downloads the data-archive files listed in shared/large_data_files.tsv into their package paths
# and checks their MD5 sums. Files already present with the right checksum are skipped.
# Run from the package root:
#
#   bash shared/scripts/download_large_data.sh          # all files
#   bash shared/scripts/download_large_data.sh Fig5     # only the files whose path starts with Fig5
set -euo pipefail

RECORD="XXXXXXX"   # Zenodo record of the data archive (DOI 10.5281/zenodo.XXXXXXX)
LIST="shared/large_data_files.tsv"
PREFIX="${1:-}"

[ -f "$LIST" ] || { echo "run from the package root ($LIST not found)" >&2; exit 1; }
[ "$RECORD" != "XXXXXXX" ] || { echo "set RECORD to the Zenodo record of the data archive" >&2; exit 1; }

md5_of() { if command -v md5sum > /dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }

tail -n +2 "$LIST" | while IFS=$'\t' read -r path bytes md5 panels; do
  case "$path" in "$PREFIX"*) ;; *) continue ;; esac
  if [ -f "$path" ] && [ "$(md5_of "$path")" = "$md5" ]; then
    echo "present  $path"
    continue
  fi
  mkdir -p "$(dirname "$path")"
  echo "download $path ($bytes bytes; $panels)"
  curl -fL --retry 3 -o "$path.part" "https://zenodo.org/records/$RECORD/files/$(basename "$path")?download=1"
  [ "$(md5_of "$path.part")" = "$md5" ] || { echo "MD5 mismatch: $path" >&2; rm -f "$path.part"; exit 1; }
  mv "$path.part" "$path"
done
