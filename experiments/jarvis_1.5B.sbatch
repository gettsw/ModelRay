#!/bin/bash
#SBATCH --job-name=modelray-1.5B
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=modelray_1.5B_%j.log
# ModelRay A+D confirmation run (experiments/DESIGN.md): Qwen2.5-1.5B-Instruct, phases 1+2, 48 checkpoints.
# Resumable: resubmit the same script and finished checkpoints are skipped.
# Run from the unpacked bundle dir:  sbatch experiments/jarvis_1.5B.sbatch

module load python/3.11.10
unset PYTHONHOME
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=$HOME/modelray-env
PY=$ENV/bin/python

if [ ! -x "$PY" ]; then
  /apps/python/3.11.10/bin/python3.11 -m venv "$ENV"
  $PY -m pip install -q --upgrade pip
  # Same versions as the local 0.5B run.
  $PY -m pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 \
    || $PY -m pip install -q torch==2.14.1
  $PY -m pip install -q -e . transformer_lens==4.3.0 transformers==5.22.1 peft==0.21.2 accelerate datasets pytest
else
  $PY -m pip install -q --no-deps -e .   # pick up this bundle's model_ray
fi

$PY -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
$PY -m pytest -q tests 2>&1 | tail -1
(cd experiments/data && sha256sum -c ../data_manifest.sha256)   # identical prompts to the 0.5B run

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
M=Qwen/Qwen2.5-1.5B-Instruct
$PY experiments/grid.py --model $M --phase 1 --delete-models
$PY experiments/grid.py --model $M --phase 2 --delete-models
$PY experiments/analyze.py --model Qwen2.5-1.5B-Instruct --gate v2 --subsample 50 > experiments/runs/analysis_1.5B_v2.txt
$PY experiments/analyze.py --model Qwen2.5-1.5B-Instruct --gate v1 --subsample 0 > experiments/runs/analysis_1.5B_v1.txt
tar czf ~/modelray_1.5B_results.tar.gz experiments/runs
echo "DONE: ~/modelray_1.5B_results.tar.gz"
