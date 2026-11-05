"""Run the whole DESIGN.md grid for one base model; skips cells already done (resumable).

    python experiments/grid.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
GRID = [("benign_domain", [50, 200, 800]), ("instruction", [50, 200, 800]),
        ("identity_shift", [10, 30, 100]), ("harmful", [5, 20, 100])]
MIXED = [0.01, 0.05, 0.10]  # harmful share at 200 steps


def run(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--delete-models", action="store_true",
                help="delete each merged checkpoint once evaluated (cluster quota; keeps harmful weights off shared disk)")
ap.add_argument("--phase", type=int, default=1, help="2 = the negatives/lead-time cells added after phase 1")
a = ap.parse_args()
short = a.model.split("/")[-1]
run(str(HERE / "evaluate.py"), "--base", a.model)
cells = [(c, s, f"{c}__s{s}", []) for c, doses in GRID for s in doses]
cells += [("mixed", 200, f"mixed{f:g}__s200", ["--harmful-frac", str(f)]) for f in MIXED]
if a.phase == 2:  # see DESIGN.md change log: phase 1 produced no intact checkpoints
    cells = [("mixed", 200, f"benign_domain+refusal{f:g}__s200", ["--mix-base", "benign_domain", "--refusal-frac", str(f)])
             for f in (0.03, 0.10)]
    cells += [("mixed", 200, "instruction+refusal0.1__s200", ["--mix-base", "instruction", "--refusal-frac", "0.1"])]
    cells += [("refusal", 20, "refusal__s20", [])]
    cells += [("benign_domain", s, f"benign_domain-lr1e-05__s{s}", ["--lr", "1e-5"]) for s in (20, 50, 200)]
    cells += [("harmful", s, f"harmful__s{s}", []) for s in (1, 2)]
for seed in a.seeds:
    for cond, steps, tag, extra in cells:
        name = f"{short}__{tag}__seed{seed}"
        if (HERE / "runs" / name / "eval.json").exists():  # resumable even after the weights are deleted
            continue
        run(str(HERE / "train.py"), "--model", a.model, "--condition", cond, "--steps", str(steps),
            "--seed", str(seed), *extra)
        run(str(HERE / "evaluate.py"), "--base", a.model, "--run", name)
        if a.delete_models:
            shutil.rmtree(HERE / "runs" / name / "model")
