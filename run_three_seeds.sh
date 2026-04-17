#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

SEEDS=(42 7 21)

for s in "${SEEDS[@]}"; do
  echo "Running seed ${s}"
  python finetune_forecasting.py \
    --train-from-scratch \
    --pooling flatten \
    --linear-probe-epochs 10 \
    --finetune-epochs 25 \
    --backbone-lr 1e-5 \
    --head-lr 5e-4 \
    --backbone-warmup-epochs 3 \
    --batch-size 16 \
    --seed "${s}"

  cp finetuning_results/results.json "finetuning_results/results_seed${s}.json"
done

python - <<'PY'
import json
import pathlib
import statistics

root = pathlib.Path("finetuning_results")
seeds = [42, 7, 21]
all_data = {}
for s in seeds:
    p = root / f"results_seed{s}.json"
    all_data[s] = json.loads(p.read_text())

summary = {}
for dataset in all_data[seeds[0]].keys():
    improvements = []
    per_seed = []
    for s in seeds:
        metrics = all_data[s][dataset]
        pre = metrics["test_mse"]
        scratch = metrics["scratch_test_mse"]
        imp = (scratch - pre) / scratch * 100.0
        improvements.append(imp)
        per_seed.append(
            {
                "seed": s,
                "mse_improvement_pct": imp,
                "pre_mse": pre,
                "scratch_mse": scratch,
            }
        )

    summary[dataset] = {
        "mean_mse_improvement_pct": sum(improvements) / len(improvements),
        "median_mse_improvement_pct": statistics.median(improvements),
        "all_positive": all(v > 0 for v in improvements),
        "per_seed": per_seed,
    }

all_improvements = [
    (all_data[s][d]["scratch_test_mse"] - all_data[s][d]["test_mse"])
    / all_data[s][d]["scratch_test_mse"]
    * 100.0
    for s in seeds
    for d in all_data[s].keys()
]

out = {
    "seeds": seeds,
    "dataset_summary": summary,
    "overall_mean_mse_improvement_pct": sum(all_improvements) / len(all_improvements),
    "overall_median_mse_improvement_pct": statistics.median(all_improvements),
}

summary_path = root / "multi_seed_summary.json"
summary_path.write_text(json.dumps(out, indent=2))
print(f"Wrote {summary_path}")
PY
