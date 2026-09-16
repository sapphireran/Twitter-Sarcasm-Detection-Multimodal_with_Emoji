#!/usr/bin/env bash
# Run every personal example script from the repository root.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

python3 examples/01_dataset_preview.py
echo
python3 examples/02_lexical_cues.py
echo
python3 examples/03_emoji_vectors.py
echo
python3 examples/04_attention_walkthrough.py
echo
python3 examples/05_tfidf_baseline.py
