#!/bin/bash
# Fetch UCF101 (videos + official recognition splits) into data/ucf101/.
#
# Why UCF101: every paper in the SSv2-CIL lineage we audited (TCD ICCV'21,
# STSP ECCV'24, CSTA, ESSENTIAL ICCV'25) reports UCF101 under the same TCD
# protocol, so a number here is directly comparable to theirs -- unlike our
# 48-class SSv2 subset. It is also *static-biased* (ESSENTIAL's own term),
# which is the regime where frozen-CLIP + mean-pool heads should be strongest,
# whereas SSv2 is temporal-biased and structurally hostile to them.
# See reports/sota_positioning_brief.md Sec 1(f).
#
# Source: the official CRCV host (www.crcv.ucf.edu) was unreachable when this
# was written, so we pull the HF mirror quchenyuan/UCF101-ZIP, which carries
# the same archives in .zip form (no unrar needed) including the official
# ucfTrainTestlist. Sizes: videos 6.47 GB, splits 114 KB.
#
# Usage: ./scripts/get_ucf101.sh [--extract]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/ucf101"
BASE="https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main"

mkdir -p "$DEST"
cd "$DEST"

fetch() {  # fetch <remote-name> <local-name>
  if [ -s "$2" ]; then echo "  have $2 — skipping"; return; fi
  echo "  downloading $2 ..."
  curl -L -C - --retry 5 --retry-delay 10 --progress-bar "$BASE/$1" -o "$2"
}

echo "UCF101 -> $DEST"
fetch "UCF101TrainTestSplits-RecognitionTask.zip" "splits.zip"
fetch "UCF-101.zip" "UCF-101.zip"

if [ ! -d ucfTrainTestlist ]; then
  bsdtar -xf splits.zip
  echo "  extracted ucfTrainTestlist/"
fi

# sanity: the official recognition split is 9537 train / 3783 test over 101 classes
tr=$(wc -l < ucfTrainTestlist/trainlist01.txt | tr -d ' ')
te=$(wc -l < ucfTrainTestlist/testlist01.txt | tr -d ' ')
cl=$(wc -l < ucfTrainTestlist/classInd.txt | tr -d ' ')
echo "  split1: train=$tr test=$te classes=$cl"
[ "$tr" = "9537" ] && [ "$te" = "3783" ] && [ "$cl" = "101" ] \
  && echo "  split counts match the official release" \
  || echo "  WARNING: split counts differ from the official 9537/3783/101"

if [ "${1:-}" = "--extract" ]; then
  if [ ! -d UCF-101 ]; then
    echo "  extracting videos (this takes a few minutes) ..."
    bsdtar -xf UCF-101.zip
  fi
  n=$(find UCF-101 -name '*.avi' | wc -l | tr -d ' ')
  echo "  extracted videos: $n (expected 13320)"
fi

echo
echo "Next: extract CLIP features, then run the TCD-protocol CIL evaluation."
