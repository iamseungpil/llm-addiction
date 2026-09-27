#!/usr/bin/env bash
# Link this folder to a checkout of the paper repo, so the figure and table scripts here
# write into the paper's images/ and read its LaTeX exactly as they did when they lived there.
#
#   ./link_paper_repo.sh                      # paper repo at ../../LLM_Addiction_NMT_KOR
#   ./link_paper_repo.sh /path/to/paper/repo  # anywhere else
#
# The links (images, neurips_content_en, neurips_content, poster) are git-ignored.
set -euo pipefail
cd "$(dirname "$0")"
PAPER="${1:-../../LLM_Addiction_NMT_KOR}"
PAPER="$(cd "$PAPER" && pwd)"
[ -f "$PAPER/shared/paper_core.tex" ] || { echo "not a paper repo: $PAPER" >&2; exit 1; }
for d in images neurips_content_en neurips_content poster; do
  [ -e "$d" ] && [ ! -L "$d" ] && { echo "refusing to replace real folder: $d" >&2; exit 1; }
  ln -sfn "$PAPER/$d" "$d"
  echo "$d -> $PAPER/$d"
done
