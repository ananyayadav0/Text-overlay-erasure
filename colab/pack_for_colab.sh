#!/usr/bin/env bash
# Pack code + dataset + finished results into colab/ca6127_colab.zip for upload to Google Drive.
set -euo pipefail
builtin cd "$(dirname "$0")/.."
rm -f colab/ca6127_colab.zip
zip -qr colab/ca6127_colab.zip \
  make_violation_dataset.py requirements.txt README.md src \
  dataset/clean dataset/violated dataset/mask dataset/annotations.jsonl \
  results \
  -x "*/__pycache__/*" "*.pyc" "*.tmp" "results/.all_done"
ls -lh colab/ca6127_colab.zip
