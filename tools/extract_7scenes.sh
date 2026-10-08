#!/usr/bin/env bash
set -euo pipefail

source_dir=/data2/liusy/datasets/7scenes-original
target_dir=data/7scenes

mkdir -p "$target_dir"
for archive in "$source_dir"/*.zip; do
  unzip -qn "$archive" -d "$target_dir"
done

while IFS= read -r -d '' archive; do
  unzip -qn "$archive" -d "$(dirname "$archive")"
done < <(find "$target_dir" -type f -name 'seq-*.zip' -print0)
