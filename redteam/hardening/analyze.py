"""Summarise the hardening runs: paired bootstrap, seed spread and illustrative rank deltas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
DETAILS = HERE / "outputs" / "details"
REF = "floss"
HONEST_PREFIX = ("floss", "v_", "w_", "seed_", "template_average")


def flat(scores: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for exp, data in scores["experiments"].items():
        for method, metrics in data["results"].items():
            for metric, value in metrics.items():
                out.setdefault(method, {})[f"{exp}/{metric}"] = value
    return out


def miou(inter: np.ndarray, union: np.ndarray) -> float:
    i, u = inter.sum(0).astype(float), union.sum(0).astype(float)
    return float(np.nanmean(np.where(u > 0, i / np.maximum(u, 1), np.nan)) * 100)


def main() -> None:
    reference = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    test = flat(json.loads((RES / "test_all.json").read_text()))
    train = flat(json.loads((RES / "train_all.json").read_text()))
    keys = [f"{e}/{m}" for e, d in reference.items() for m in d["metrics"]]
    honest = [n for n in test if n.startswith(HONEST_PREFIX)]
    std = {k: float(np.std([test[n][k] for n in honest])) for k in keys}

    def rank_delta(name: str) -> float:
        total = 0.0
        for e, d in reference.items():
            for m, spec in d["metrics"].items():
                k = f"{e}/{m}"
                sign = 1.0 if spec["higher_is_better"] else -1.0
                if std[k] > 0:
                    total += d["weight"] * abs(spec["coefficient"]) * sign * (test[name][k] - test[REF][k]) / std[k]
        return total

    rng = np.random.default_rng(0)
    ref_d = dict(np.load(DETAILS / f"{REF}.npz"))
    n_img = ref_d["inter"].shape[0]
    boots = rng.integers(0, n_img, size=(1000, n_img))
    rows = {}
    for name in test:
        row = {"test": test[name], "train": train.get(name, {}), "rank_delta_vs_floss": rank_delta(name)}
        path = DETAILS / f"{name}.npz"
        if path.exists() and name != REF:
            d = dict(np.load(path))
            diffs = np.array([miou(d["inter"][b], d["union"][b]) - miou(ref_d["inter"][b], ref_d["union"][b])
                              for b in boots])
            low_diff = d["low_miou"] - ref_d["low_miou"]
            row["core_miou_paired_bootstrap"] = {"mean": float(diffs.mean()), "std": float(diffs.std()),
                                                 "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
            row["low_data_paired_runs"] = {"mean_diff": float(low_diff.mean()),
                                           "se": float(low_diff.std(ddof=1) / np.sqrt(len(low_diff))),
                                           "runs_better": int((low_diff > 0).sum()), "runs": int(len(low_diff))}
        rows[name] = row
    seeds = [n for n in test if n.startswith("seed_pool50")]
    seed_spread = {k: {"values": [test[n][k] for n in seeds],
                       "range": float(np.ptp([test[n][k] for n in seeds])) if seeds else None} for k in keys}
    ref_low = ref_d["low_miou"]
    summary = {
        "rank_definition": "sum_exp weight * sum_metric |coef| * dir * (value - floss) / std over honest cohort (illustrative)",
        "cohort_std": std,
        "reference_low_data_runs": ref_low.tolist(),
        "seed_spread_test": seed_spread,
        "methods": rows,
    }
    for scen in ("scenario_label_tamper", "scenario_oracle_tamper", "scenario_stale"):
        p = RES / f"{scen}.json"
        if p.exists():
            summary[scen] = {"scores": flat(json.loads(p.read_text())),
                             "log_tail": (RES / f"{scen}.log").read_text().splitlines()[-4:]}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["| method | test core mIoU | test low mIoU | test expert % | train core | train low | train expert % | core Δ vs floss (bootstrap 95% CI) | low Δ (runs better /12) | ΔR |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for name, row in sorted(rows.items(), key=lambda kv: -kv[1]["rank_delta_vs_floss"]):
        t, tr = row["test"], row["train"]
        b = row.get("core_miou_paired_bootstrap")
        lo = row.get("low_data_paired_runs")
        lines.append("| {} | {:.2f} | {:.2f} | {:.1f} | {:.2f} | {:.2f} | {:.1f} | {} | {} | {:+.3f} |".format(
            name, *(t.get(k, float("nan")) for k in keys), *(tr.get(k, float("nan")) for k in keys),
            f"{b['mean']:+.2f} [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]" if b else "—",
            f"{lo['mean_diff']:+.2f} ({lo['runs_better']}/{lo['runs']})" if lo else "—",
            row["rank_delta_vs_floss"]))
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"seed_spread_test": seed_spread, "cohort_std": std}, indent=2))


if __name__ == "__main__":
    main()
