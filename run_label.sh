#!/bin/bash
set -euo pipefail

SHARD_ID="${1:-0}"
NUM_SHARDS="${NUM_SHARDS:-10}"

INPUT="${INPUT_JSONL:-filtered_pairs_10k.jsonl}"
OUTDIR="."
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"

mkdir -p "$OUTDIR"

echo "[env] hostname=$(hostname)"
echo "[env] shard=$SHARD_ID num_shards=$NUM_SHARDS model=$MODEL_NAME"
echo "[env] input=$INPUT outdir=$OUTDIR"

export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
mkdir -p "$HF_HOME"

export PYTHONNOUSERSITE=1

IMG="docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime"

apptainer exec --nv --cleanenv \
  --env HF_HOME="$HF_HOME" \
  --env PYTHONNOUSERSITE=1 \
  --env TRANSFORMERS_NO_TORCHVISION=1 \
  "$IMG" \
  bash -lc "
    set -euo pipefail
    python3 -V
    python3 -c 'import torch; print(\"torch(base):\", torch.__version__)'

    # Fresh writable package dir every run (prevents leftovers)
    PKGDIR=\$(mktemp -d -p \$PWD py_pkgs_XXXXXX)

    # Install ONLY what we need, with --no-deps so pip never pulls torch
    python3 -m pip -q install --no-cache-dir --no-deps --target \"\$PKGDIR\" \
      transformers==4.48.0 \
      tokenizers==0.21.4 \
      huggingface-hub==0.27.1 \
      safetensors==0.4.5 \
      accelerate==1.2.1 \
      regex==2024.11.6 \
      packaging==24.2 \
      pyyaml==6.0.2 \
      requests==2.32.3 \
      tqdm==4.67.1 \
      filelock==3.16.1 \
      numpy==1.26.4

    export PYTHONPATH=\"\$PKGDIR:\${PYTHONPATH:-}\"

    # Sanity checks:
    python3 - <<'PY'
import os, sys
import torch
print('torch(using):', torch.__version__)
# ensure we're NOT importing torch from PKGDIR
import torch as t
print('torch(file):', t.__file__)
import transformers, tokenizers
print('transformers:', transformers.__version__)
print('tokenizers:', tokenizers.__version__)
print('TRANSFORMERS_NO_TORCHVISION:', os.environ.get('TRANSFORMERS_NO_TORCHVISION'))
PY

    python3 -u label_worker.py \
      --in '$INPUT' \
      --out '$OUTDIR/labeled_${SHARD_ID}.jsonl' \
      --shard-id '$SHARD_ID' \
      --num-shards '$NUM_SHARDS' \
      --model '$MODEL_NAME' \
      --topk 5 \
      --max-new-tokens 220 \
      --qwen-min-conf 0.55 \
      --max-rows 20
  "

