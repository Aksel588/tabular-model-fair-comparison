"""
Post-hoc analysis: competitiveness vs best, overfitting, complexity proxies.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import config


def _load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def complexity_rank(model: str) -> int:
    """Ordinal complexity proxy (higher = more flexible / typically heavier)."""
    order = {
        "logistic_regression": 1,
        "ridge": 1,
        "decision_tree": 2,
        "random_forest": 3,
        "xgboost": 4,
    }
    return order.get(model, 0)


def analyze_block(block: Dict[str, Any]) -> Dict[str, Any]:
    task = block["dataset_meta"]["task"]
    ds = block["dataset_meta"]["key"]
    results = block["results"]
    baselines = block["baselines"]

    if task == "classification":
        best_f1 = max(r["test_f1"] for r in results)
        competitive = []
        for r in results:
            margin = best_f1 - r["test_f1"]
            comp = margin <= config.COMPETITIVE_MARGIN_F1
            competitive.append(
                {
                    "model": r["model"],
                    "test_f1": r["test_f1"],
                    "within_margin_of_best": comp,
                    "margin_to_best_f1": margin,
                    "gap_f1_train_val": r["gap_f1"],
                    "complexity_rank": complexity_rank(r["model"]),
                    "fit_time_s": r["fit_time_s"],
                }
            )
        simple = [c for c in competitive if c["model"] in ("logistic_regression", "decision_tree")]
        complex_m = [c for c in competitive if c["model"] in ("random_forest", "xgboost")]
        return {
            "dataset": ds,
            "task": task,
            "baseline_f1": baselines.get("baseline_majority_f1"),
            "best_test_f1": best_f1,
            "competitive_table": sorted(competitive, key=lambda x: -x["test_f1"]),
            "simple_models_competitive": [x["model"] for x in simple if x["within_margin_of_best"]],
            "complex_models_competitive": [x["model"] for x in complex_m if x["within_margin_of_best"]],
            "overfitting_leader": max(results, key=lambda r: r["gap_f1"])["model"],
            "max_gap_f1": max(r["gap_f1"] for r in results),
        }

    best_rmse = min(r["test_rmse"] for r in results)
    competitive = []
    for r in results:
        rel = (r["test_rmse"] - best_rmse) / max(best_rmse, 1e-9)
        comp = rel <= config.COMPETITIVE_MARGIN_RMSE_REL
        competitive.append(
            {
                "model": r["model"],
                "test_rmse": r["test_rmse"],
                "within_relative_margin_of_best": comp,
                "relative_gap_to_best": rel,
                "gap_rmse_train_test": r["gap_rmse"],
                "complexity_rank": complexity_rank(r["model"]),
                "fit_time_s": r["fit_time_s"],
            }
        )
    simple = [c for c in competitive if c["model"] in ("ridge", "decision_tree")]
    complex_m = [c for c in competitive if c["model"] in ("random_forest", "xgboost")]
    return {
        "dataset": ds,
        "task": task,
        "baseline_rmse": baselines.get("baseline_mean_rmse"),
        "best_test_rmse": best_rmse,
        "competitive_table": sorted(competitive, key=lambda x: x["test_rmse"]),
        "simple_models_competitive": [x["model"] for x in simple if x["within_relative_margin_of_best"]],
        "complex_models_competitive": [x["model"] for x in complex_m if x["within_relative_margin_of_best"]],
        "overfitting_leader": max(results, key=lambda r: r["gap_rmse"])["model"],
        "max_gap_rmse": max(r["gap_rmse"] for r in results),
    }


def write_analysis_report(blocks: List[Dict[str, Any]], out_path: str) -> None:
    lines = [
        "# Analysis: simple vs complex models",
        "",
        f"Protocol: competitive F1 margin ≤ {config.COMPETITIVE_MARGIN_F1}; "
        f"competitive RMSE relative margin ≤ {config.COMPETITIVE_MARGIN_RMSE_REL:.0%}.",
        "",
    ]
    for block in blocks:
        a = analyze_block(block)
        lines.append(f"## {a['dataset']} ({a['task']})")
        lines.append("")
        if a["task"] == "classification":
            lines.append(f"- Baseline majority F1: {a['baseline_f1']:.4f}")
            lines.append(f"- Best test F1: {a['best_test_f1']:.4f}")
            lines.append(f"- Largest train–test F1 gap (model): **{a['overfitting_leader']}** (max gap {a['max_gap_f1']:.4f})")
            sm = ", ".join(a["simple_models_competitive"]) or "(none)"
            cm = ", ".join(a["complex_models_competitive"]) or "(none)"
            lines.append("- Competitive simple models: " + sm)
            lines.append("- Competitive complex models: " + cm)
            lines.append("")
            lines.append("| model | test F1 | within margin | train–test gap | fit (s) |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in a["competitive_table"]:
                lines.append(
                    f"| {row['model']} | {row['test_f1']:.4f} | {row['within_margin_of_best']} | "
                    f"{row['gap_f1_train_val']:.4f} | {row['fit_time_s']:.2f} |"
                )
        else:
            lines.append(f"- Baseline mean RMSE: {a['baseline_rmse']:.4f}")
            lines.append(f"- Best test RMSE: {a['best_test_rmse']:.4f}")
            lines.append(f"- Largest train–test RMSE gap (model): **{a['overfitting_leader']}** (max gap {a['max_gap_rmse']:.4f})")
            sm = ", ".join(a["simple_models_competitive"]) or "(none)"
            cm = ", ".join(a["complex_models_competitive"]) or "(none)"
            lines.append("- Competitive simple models: " + sm)
            lines.append("- Competitive complex models: " + cm)
            lines.append("")
            lines.append("| model | test RMSE | within margin | train–test gap | fit (s) |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in a["competitive_table"]:
                lines.append(
                    f"| {row['model']} | {row['test_rmse']:.4f} | {row['within_relative_margin_of_best']} | "
                    f"{row['gap_rmse_train_test']:.4f} | {row['fit_time_s']:.2f} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Interpretation notes",
            "",
            "- **When simple models work well**: near-linear decision boundaries, strong numeric features, ",
            "or limited sample size where deep trees overfit (watch train–test gap).",
            "- **When complex models pull ahead**: non-additive interactions, heterogeneous subpopulations, ",
            "or smooth nonlinear structure (housing) where ensembles reduce variance.",
            "- **Overfitting**: large positive F1 gap or RMSE gap indicates memorization; compare against CV score.",
            "- **Tradeoff**: higher complexity rank vs. marginal test gain and `fit_time_s` / inference ms.",
            "",
        ]
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_complexity_tradeoff(all_results_path: str, fig_path: str) -> None:
    """Scatter: complexity rank vs test metric; one series per dataset."""
    import matplotlib.pyplot as plt

    blocks = _load_results(all_results_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_cls, ax_reg = axes
    seen_cls: set[str] = set()
    seen_reg: set[str] = set()
    for block in blocks:
        meta = block["dataset_meta"]
        name = meta["key"]
        for r in block["results"]:
            cr = complexity_rank(r["model"])
            if meta["task"] == "classification":
                label = name if name not in seen_cls else None
                seen_cls.add(name)
                ax_cls.scatter(cr, r["test_f1"], label=label)
            else:
                label = name if name not in seen_reg else None
                seen_reg.add(name)
                ax_reg.scatter(cr, r["test_rmse"], label=label)
    ax_cls.set_xlabel("Complexity rank (ordinal)")
    ax_cls.set_ylabel("Test F1 (weighted)")
    ax_cls.set_title("Classification")
    ax_cls.legend()
    ax_reg.set_xlabel("Complexity rank (ordinal)")
    ax_reg.set_ylabel("Test RMSE (lower better)")
    ax_reg.set_title("Regression")
    ax_reg.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
