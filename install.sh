#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
bin_dir=${MATH2MD_BIN_DIR:-"$HOME/.local/bin"}
mkdir -p "$bin_dir"

ln -sfn "$repo_dir/math2md/math2md" "$bin_dir/math2md"
ln -sfn "$repo_dir/board2md/board2md" "$bin_dir/board2md"
ln -sfn "$repo_dir/video2md_review/video2md" "$bin_dir/video2md"
ln -sfn "$repo_dir/video2md_review/video2md-merge" "$bin_dir/video2md-merge"
ln -sfn "$repo_dir/video2md_review/v2md-queue2" "$bin_dir/v2md-queue2"
ln -sfn "$repo_dir/video2md_review/v2md-status" "$bin_dir/v2md-status"

printf 'Installed Math2MD commands in %s\n' "$bin_dir"
printf 'Ensure that directory is on PATH before running math2md.\n'
