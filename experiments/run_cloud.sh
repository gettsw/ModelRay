#!/bin/bash
# ModelRay 1.5B confirmation run on a single big GPU box (AMD Developer Cloud MI300X / ROCm, or any CUDA box).
# No SLURM. Resumable: re-run the script and finished checkpoints are skipped.
#   cd ~/modelray && nohup bash experiments/run_cloud.sh > run_cloud.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=$HOME/modelray-env
PY=$ENV/bin/python

if [ ! -x "$PY" ]; then
  # Reuse a preinstalled GPU torch (ROCm images ship one) if present, else install the ROCm wheel.
  if python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python3 -m venv --system-site-packages "$ENV"
  else
    python3 -m venv "$ENV"
    $PY -m pip install -q --upgrade pip
    $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/rocm6.4
  fi
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q -e . transformer_lens==4.3.0 transformers==5.22.1 peft==0.21.2 accelerate datasets pytest
fi

$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'hip', torch.version.hip, torch.cuda.get_device_name(0))"
$PY -m pytest -q tests 2>&1 | tail -1
(cd experiments/data && sha256sum -c ../data_manifest.sha256)   # identical prompts to the 0.5B run

M=Qwen/Qwen2.5-1.5B-Instruct
$PY experiments/evaluate.py --base $M                            # base + layer sweep, once, before the workers
# 192 GB is plenty for 4 concurrent 1.5B workers; each grid cell belongs to exactly one worker.
pids=()
for phase in 1 2; do
  for seed in 0 1; do
    $PY experiments/grid.py --model $M --phase $phase --seeds $seed --delete-models \
      > experiments/runs/grid_1.5B_p${phase}_s${seed}.log 2>&1 &
    pids+=($!)
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] || { echo "a worker failed; see experiments/runs/grid_1.5B_*.log, then re-run this script"; exit 1; }

$PY experiments/analyze.py --model Qwen2.5-1.5B-Instruct --gate v2 --subsample 50 > experiments/runs/analysis_1.5B_v2.txt
$PY experiments/analyze.py --model Qwen2.5-1.5B-Instruct --gate v1 --subsample 0 > experiments/runs/analysis_1.5B_v1.txt
tar czf ~/modelray_1.5B_results.tar.gz experiments/runs
echo "DONE: ~/modelray_1.5B_results.tar.gz"
