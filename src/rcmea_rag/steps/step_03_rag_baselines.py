#!/usr/bin/env python3
"""Plain RAG and threshold baselines for RCMEA-RAG.

This script freezes the reviewer-facing "ordinary RAG" comparisons:
- weighted retrieval evidence with a single train-selected threshold;
- ROI-as-positive evidence variants;
- simple route-vote variants.

All thresholds/configurations are selected on training cases only. evaluation is
used only for fixed evaluation of the train-selected rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_ROOT = Path(os.environ.get("RCMEA_OUTPUT_ROOT", "outputs")).resolve()
OUT_DIR = OUTPUT_ROOT / "step_03_rag_baselines"

from . import step_01_policy_search as comp
from . import step_02_roi_route as focused


MAIN_SCENARIO = "default_training_bank__balanced"
RECALL_FLOORS = [0.90, 0.875, 0.85, 0.825, 0.80, 0.775, 0.75]
SPEC_FLOORS = [0.85, 0.825, 0.80, 0.775, 0.75]
FP_RATE_BUDGETS = [0.12, 0.15, 0.175, 0.20, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--scenario", default=MAIN_SCENARIO)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--candidate-pool", type=int, default=80)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cfg_json(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def metric_prefix(prefix: str, metric: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metric.items()}


def route_idx(routes: list[str]) -> list[int]:
    return [comp.ROUTE_INDEX[route] for route in routes]


def count_str(row: dict[str, Any], prefix: str) -> str:
    return "/".join(str(int(round(float(row[f"{prefix}_{key}"])))) for key in ["tp", "tn", "fp", "fn"])


def base_baselines() -> list[dict[str, Any]]:
    """Return baseline families before threshold expansion."""
    return [
        {
            "baseline": "trunk_only",
            "family": "weighted_plain_rag",
            "kind": "weighted_margin",
            "routes": ["fullct", "rsuper", "radgpt"],
            "weights": {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75},
            "description": "three trunk routes with one train-selected threshold",
        },
        {
            "baseline": "fullct_only",
            "family": "single_route",
            "kind": "weighted_margin",
            "routes": ["fullct"],
            "weights": {"fullct": 1.0},
            "description": "single full CT retrieval route",
        },
        {
            "baseline": "rsuper_only",
            "family": "single_route",
            "kind": "weighted_margin",
            "routes": ["rsuper"],
            "weights": {"rsuper": 1.0},
            "description": "single clinical-neighbor route",
        },
        {
            "baseline": "radgpt_only",
            "family": "single_route",
            "kind": "weighted_margin",
            "routes": ["radgpt"],
            "weights": {"radgpt": 1.0},
            "description": "single structured-neighbor route",
        },
        {
            "baseline": "merlinroi_only",
            "family": "single_route",
            "kind": "weighted_margin",
            "routes": ["merlinroi_bal"],
            "weights": {"merlinroi_bal": 1.0},
            "description": "single type-balanced Merlin ROI route",
        },
        {
            "baseline": "trunk_merlin_positive_equal",
            "family": "roi_as_positive_plain_rag",
            "kind": "weighted_margin",
            "routes": ["fullct", "rsuper", "radgpt", "merlinroi_bal"],
            "weights": {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75, "merlinroi_bal": 1.0},
            "description": "Merlin ROI added as ordinary positive evidence",
        },
        {
            "baseline": "trunk_merlin_positive_light",
            "family": "roi_as_positive_plain_rag",
            "kind": "weighted_margin",
            "routes": ["fullct", "rsuper", "radgpt", "merlinroi_bal"],
            "weights": {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75, "merlinroi_bal": 0.25},
            "description": "Merlin ROI added as lightly weighted positive evidence",
        },
        {
            "baseline": "all_routes",
            "family": "weighted_plain_rag",
            "kind": "weighted_margin",
            "routes": comp.ROUTES,
            "weights": {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75, "merlinroi_bal": 0.25},
            "description": "all routes as ordinary weighted positive evidence",
        },
        {
            "baseline": "all_routes_roi_balanced",
            "family": "weighted_plain_rag",
            "kind": "weighted_margin",
            "routes": comp.ROUTES,
            "weights": {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75, "merlinroi_bal": 0.75},
            "description": "all routes with balanced ROI weights",
        },
        {
            "baseline": "trunk_vote",
            "family": "vote_plain_rag",
            "kind": "route_vote",
            "routes": ["fullct", "rsuper", "radgpt"],
            "weights": {},
            "description": "present if enough trunk routes vote present",
        },
        {
            "baseline": "all_routes_vote",
            "family": "vote_plain_rag",
            "kind": "route_vote",
            "routes": comp.ROUTES,
            "weights": {},
            "description": "present if enough of all routes vote present",
        },
        {
            "baseline": "trunk_merlin_vote",
            "family": "vote_plain_rag",
            "kind": "route_vote",
            "routes": ["fullct", "rsuper", "radgpt", "merlinroi_bal"],
            "weights": {},
            "description": "present if enough trunk plus Merlin ROI routes vote present",
        },
    ]


def candidate_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = list(comp.THRESHOLDS)

    def add(cfg: dict[str, Any]) -> None:
        row = dict(cfg)
        row["config_id"] = f"PRB{len(rows) + 1:05d}"
        rows.append(row)

    for base in base_baselines():
        if base["kind"] == "weighted_margin":
            for threshold in thresholds:
                add({**base, "threshold": float(threshold), "min_pos_routes": ""})
        elif base["kind"] == "route_vote":
            for min_pos_routes in range(1, len(base["routes"]) + 1):
                add({**base, "threshold": "", "min_pos_routes": int(min_pos_routes)})
        else:
            raise ValueError(base["kind"])
    return rows


def weighted_score(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    idx = route_idx(cfg["routes"])
    weights = np.asarray([float(cfg["weights"].get(route, 0.0)) for route in comp.ROUTES], dtype=np.float32)
    return arr["margins"][:, idx] @ weights[idx]


def predict_config(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if cfg["kind"] == "weighted_margin":
        score = weighted_score(arr, cfg)
        pred = score > float(cfg["threshold"])
        return pred, {
            "score_mean": float(np.mean(score)),
            "present_pred_n": int(np.sum(pred)),
            "threshold": float(cfg["threshold"]),
        }
    if cfg["kind"] == "route_vote":
        idx = route_idx(cfg["routes"])
        votes = np.sum(arr["margins"][:, idx] > 0, axis=1)
        pred = votes >= int(cfg["min_pos_routes"])
        return pred, {
            "score_mean": float(np.mean(votes)),
            "present_pred_n": int(np.sum(pred)),
            "min_pos_routes": int(cfg["min_pos_routes"]),
        }
    raise ValueError(cfg["kind"])


def best_f1_key(metric: dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        float(metric["f1"]),
        float(metric["balanced_accuracy"]),
        float(metric["specificity"]),
        float(metric["precision"]),
        -float(metric["fp"]),
    )


def min_fp_key(metric: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    return (
        -float(metric["fp"]),
        float(metric["specificity"]),
        float(metric["precision"]),
        float(metric["f1"]),
        float(metric["recall"]),
        -float(metric["fn"]),
    )


def max_recall_key(metric: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metric["recall"]),
        float(metric["f1"]),
        float(metric["specificity"]),
        -float(metric["fp"]),
    )


def update_selected(
    selected: dict[tuple[str, str], dict[str, Any]],
    scope: str,
    objective: str,
    cfg: dict[str, Any],
    metric: dict[str, float],
    extra: dict[str, Any],
    mode: str,
) -> None:
    key = (scope, objective)
    current = selected.get(key)
    if mode == "best_f1":
        better = current is None or best_f1_key(metric) > best_f1_key(current["metric"])
    elif mode == "min_fp":
        better = current is None or min_fp_key(metric) > min_fp_key(current["metric"])
    elif mode == "max_recall":
        better = current is None or max_recall_key(metric) > max_recall_key(current["metric"])
    else:
        raise ValueError(mode)
    if better:
        selected[key] = {"cfg": cfg, "metric": metric, "extra": extra}


def train_metric_rows(
    arr: dict[str, np.ndarray],
    configs: list[dict[str, Any]],
    sample_weight: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = arr["gold_present"]
    for cfg in configs:
        pred, extra = predict_config(arr, cfg)
        metric = comp.metric_from_bool(y, pred, sample_weight)
        neg_total = float(metric["tn"] + metric["fp"])
        fp_rate = float(metric["fp"]) / neg_total if neg_total else 0.0
        rows.append(
            {
                "config_id": cfg["config_id"],
                "baseline": cfg["baseline"],
                "family": cfg["family"],
                "kind": cfg["kind"],
                "routes": "+".join(cfg["routes"]),
                "description": cfg["description"],
                "threshold": cfg["threshold"],
                "min_pos_routes": cfg["min_pos_routes"],
                "config_json": cfg_json(cfg),
                **metric_prefix("train", metric),
                "train_fp_rate": fp_rate,
                **{f"train_{key}": value for key, value in extra.items()},
            }
        )
    return rows


def select_on_train(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in train_rows:
        cfg = json.loads(row["config_json"])
        metric = {key.removeprefix("train_"): row[key] for key in row if key.startswith("train_")}
        metric = {key: float(value) for key, value in metric.items() if key in {"tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy"}}
        extra = {key.removeprefix("train_"): row[key] for key in row if key.startswith("train_") and key not in {f"train_{m}" for m in metric}}
        scopes = ["global", cfg["family"], cfg["baseline"]]
        for scope in scopes:
            update_selected(selected, scope, "best_f1", cfg, metric, extra, "best_f1")
            for recall in RECALL_FLOORS:
                if float(metric["recall"]) >= recall:
                    update_selected(selected, scope, f"min_fp_recall_ge_{recall:.3f}".replace(".", "p"), cfg, metric, extra, "min_fp")
            for spec in SPEC_FLOORS:
                if float(metric["specificity"]) >= spec:
                    update_selected(selected, scope, f"max_recall_spec_ge_{spec:.3f}".replace(".", "p"), cfg, metric, extra, "max_recall")
            for budget in FP_RATE_BUDGETS:
                if float(row["train_fp_rate"]) <= budget:
                    update_selected(selected, scope, f"max_recall_fpr_le_{budget:.3f}".replace(".", "p"), cfg, metric, extra, "max_recall")
            if float(metric["recall"]) >= 0.85 and float(row["train_fp_rate"]) <= 0.225:
                update_selected(selected, scope, "target_train_recall_ge_0p85_fpr_le_0p225", cfg, metric, extra, "best_f1")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for (scope, objective), item in selected.items():
        cfg = item["cfg"]
        key = (scope, objective, cfg_json(cfg))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "selection_scope": scope,
                "objective": objective,
                "config_id": cfg["config_id"],
                "baseline": cfg["baseline"],
                "family": cfg["family"],
                "kind": cfg["kind"],
                "routes": "+".join(cfg["routes"]),
                "description": cfg["description"],
                "threshold": cfg["threshold"],
                "min_pos_routes": cfg["min_pos_routes"],
                "config_json": cfg_json(cfg),
                **metric_prefix("train", item["metric"]),
                **{f"train_{key}": value for key, value in item["extra"].items()},
            }
        )
    return rows


def evaluate_config_rows(rows: list[dict[str, Any]], final_arr: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cfg = json.loads(row["config_json"])
        pred, extra = predict_config(final_arr, cfg)
        metric = comp.metric_from_bool(final_arr["gold_present"], pred)
        out.append({**row, **metric_prefix("evaluation", metric), **{f"evaluation_{key}": value for key, value in extra.items()}})
    return out


def grid_rows(
    train_rows: list[dict[str, Any]],
    final_arr: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in train_rows:
        cfg = json.loads(row["config_json"])
        pred, extra = predict_config(final_arr, cfg)
        metric = comp.metric_from_bool(final_arr["gold_present"], pred)
        out.append({**row, **metric_prefix("evaluation", metric), **{f"evaluation_{key}": value for key, value in extra.items()}})
    return out


def best_from(rows: list[dict[str, Any]], *, scope: str | None = None, baseline: str | None = None, family: str | None = None, objective: str | None = None) -> dict[str, Any] | None:
    candidates = rows
    if scope is not None:
        candidates = [row for row in candidates if row["selection_scope"] == scope]
    if baseline is not None:
        candidates = [row for row in candidates if row["baseline"] == baseline]
    if family is not None:
        candidates = [row for row in candidates if row["family"] == family]
    if objective is not None:
        candidates = [row for row in candidates if row["objective"] == objective]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float(row["train_f1"]), float(row["train_balanced_accuracy"]), float(row["train_specificity"]), -float(row["train_fp"])))


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        (
            "Global train-selected plain RAG",
            {"scope": "global", "objective": "best_f1"},
            "best by train F1 across ordinary RAG baselines",
        ),
        ("Trunk-only vanilla RAG", {"baseline": "trunk_only", "objective": "best_f1"}, "same trunk routes, no ROI risk control"),
        ("All-route vanilla RAG", {"baseline": "all_routes", "objective": "best_f1"}, "all evidence routes as ordinary positive evidence"),
        ("Balanced all-route vanilla RAG", {"baseline": "all_routes_roi_balanced", "objective": "best_f1"}, "all routes with larger ROI weights"),
        ("Merlin-as-positive RAG", {"baseline": "trunk_merlin_positive_equal", "objective": "best_f1"}, "Merlin ROI used as a fourth positive trunk"),
        ("Best route-vote RAG", {"family": "vote_plain_rag", "objective": "best_f1"}, "simple majority/vote baseline"),
        ("Best single-route baseline", {"family": "single_route", "objective": "best_f1"}, "best individual retrieval stream"),
        (
            "Plain RAG target recall/FPR",
            {"scope": "global", "objective": "target_train_recall_ge_0p85_fpr_le_0p225"},
            "train-selected low-risk operating point",
        ),
        (
            "Plain RAG low-FP recall>=0.85",
            {"scope": "global", "objective": "min_fp_recall_ge_0p850"},
            "train-selected low-FP point under recall floor",
        ),
        (
            "Plain RAG max recall spec>=0.80",
            {"scope": "global", "objective": "max_recall_spec_ge_0p800"},
            "train-selected specificity-constrained point",
        ),
    ]
    out: list[dict[str, Any]] = []
    for role, filters, note in specs:
        row = best_from(rows, **filters)
        if row is None:
            continue
        out.append(
            {
                "role": role,
                "baseline": row["baseline"],
                "family": row["family"],
                "kind": row["kind"],
                "selection_scope": row["selection_scope"],
                "objective": row["objective"],
                "config_id": row["config_id"],
                "routes": row["routes"],
                "threshold": row["threshold"],
                "min_pos_routes": row["min_pos_routes"],
                "evaluation_tp_tn_fp_fn": count_str(row, "evaluation"),
                "evaluation_f1": row["evaluation_f1"],
                "evaluation_specificity": row["evaluation_specificity"],
                "evaluation_recall": row["evaluation_recall"],
                "evaluation_fp": row["evaluation_fp"],
                "evaluation_fn": row["evaluation_fn"],
                "train_tp_tn_fp_fn": count_str(row, "train"),
                "train_f1": row["train_f1"],
                "train_specificity": row["train_specificity"],
                "train_recall": row["train_recall"],
                "note": note,
            }
        )
    return out


def build_markdown(path: Path, summaries: list[dict[str, Any]], selected_rows: list[dict[str, Any]], scenario_name: str) -> None:
    lines = [
        "# Plain RAG Baselines For RCMEA-RAG",
        "",
        f"Scenario: `{scenario_name}`",
        "",
        "Selection uses train-only metrics. evaluation is fixed one-shot evaluation only.",
        "",
        "Plain baselines treat retrieval evidence as direct positive evidence or route votes; they do not use ROI counter-evidence, rescue logic, conflict penalties, or post-hoc risk veto.",
        "",
        "## Summary",
        "",
        "| role | baseline | objective | final TP/TN/FP/FN | F1 | Spec | Recall | train TP/TN/FP/FN | config |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            "| {role} | {baseline} | {objective} | {counts} | {f1:.4f} | {spec:.4f} | {rec:.4f} | {train_counts} | {cfg} |".format(
                role=row["role"],
                baseline=row["baseline"],
                objective=row["objective"],
                counts=row["evaluation_tp_tn_fp_fn"],
                f1=float(row["evaluation_f1"]),
                spec=float(row["evaluation_specificity"]),
                rec=float(row["evaluation_recall"]),
                train_counts=row["train_tp_tn_fp_fn"],
                cfg=row["config_id"],
            )
        )

    best_rows = sorted(selected_rows, key=lambda row: (-float(row["evaluation_f1"]), float(row["evaluation_fp"])))[:20]
    low_fp_rows = sorted(
        [row for row in selected_rows if row["objective"].startswith("min_fp_recall_ge")],
        key=lambda row: (float(row["evaluation_fp"]), -float(row["evaluation_f1"]), float(row["evaluation_fn"])),
    )[:20]

    def add_section(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| scope | objective | baseline | final TP/TN/FP/FN | F1 | Spec | Recall | train TP/TN/FP/FN | config |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| {scope} | {objective} | {baseline} | {counts} | {f1:.4f} | {spec:.4f} | {rec:.4f} | {train_counts} | {cfg} |".format(
                    scope=row["selection_scope"],
                    objective=row["objective"],
                    baseline=row["baseline"],
                    counts=count_str(row, "evaluation"),
                    f1=float(row["evaluation_f1"]),
                    spec=float(row["evaluation_specificity"]),
                    rec=float(row["evaluation_recall"]),
                    train_counts=count_str(row, "train"),
                    cfg=row["config_id"],
                )
            )

    add_section("Top Train-Selected Rows By Final F1", best_rows)
    add_section("Low-FP Train-Selected Rows", low_fp_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_scenarios(args: argparse.Namespace) -> list[dict[str, Any]]:
    scenarios = focused.scenario_rows()
    if args.all_scenarios:
        return scenarios
    rows = [row for row in scenarios if row["scenario"] == args.scenario]
    if not rows:
        raise ValueError(f"Unknown scenario {args.scenario!r}; available: {[row['scenario'] for row in scenarios]}")
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = focused.build_inputs()
    final_gold = focused.load_final_gold(focused.FINAL_GOLD_CSV)
    configs = candidate_configs()
    scenarios = selected_scenarios(args)
    print(json.dumps({"configs": len(configs), "scenarios": [row["scenario"] for row in scenarios]}, ensure_ascii=False), flush=True)

    arrays_by_bank: dict[tuple[str, str, bool], tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]] = {}
    all_train_rows: list[dict[str, Any]] = []
    all_grid_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    all_final_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        bank = comp.bank_key(scenario)
        if bank not in arrays_by_bank:
            print(f"[bank] {bank}", flush=True)
            train_arr, _train_cov = comp.build_arrays_for_fold(
                scenario,
                None,
                inputs["train_case_ids"],
                inputs["original_gold"],
                inputs["audit_extra"],
                inputs["train_fold_ids"],
                inputs["raw_routes"],
                inputs["med_key"],
                inputs["mer_key"],
                args.embedding_batch_size,
                args.candidate_pool,
            )
            final_arr, coverage = focused.build_final_arrays_for_bank(
                scenario,
                final_gold,
                inputs,
                args.embedding_batch_size,
                args.candidate_pool,
            )
            arrays_by_bank[bank] = (train_arr, final_arr, coverage)
            coverage_rows.append(
                {
                    "bank": "|".join(str(item) for item in bank),
                    "final_cases": coverage["final_cases"],
                    "allowed_key_cases": coverage["allowed_key_cases"],
                    "extra_key_cases": coverage["extra_key_cases"],
                    **{f"{route}_cases_with_hits": coverage["route_cases_with_hits"][route] for route in comp.ROUTES},
                }
            )
        train_arr, final_arr, _coverage = arrays_by_bank[bank]
        print(f"[scenario] {scenario['scenario']}", flush=True)
        sample_weight = comp.sample_weights_for_scenario(train_arr, scenario)
        train_rows = [{**scenario, **row} for row in train_metric_rows(train_arr, configs, sample_weight)]
        selected = [{**scenario, **row} for row in select_on_train(train_rows)]
        fixed_selected = [{**scenario, **row} for row in evaluate_config_rows(selected, final_arr)]
        fixed_grid = [{**scenario, **row} for row in grid_rows(train_rows, final_arr)]
        all_train_rows.extend(train_rows)
        all_selected_rows.extend(selected)
        all_final_rows.extend(fixed_selected)
        all_grid_rows.extend(fixed_grid)

    summaries = summary_rows(all_final_rows)
    write_csv(args.output_dir / "plain_rag_train_grid.csv", all_train_rows)
    write_csv(args.output_dir / "plain_rag_threshold_grid.csv", all_grid_rows)
    write_csv(args.output_dir / "plain_rag_train_selected_configs.csv", all_selected_rows)
    write_csv(args.output_dir / "plain_rag_evaluation_fixed_eval.csv", all_final_rows)
    write_csv(args.output_dir / "plain_rag_baseline_summary.csv", summaries)
    write_csv(args.output_dir / "coverage_by_bank.csv", coverage_rows)
    build_markdown(args.output_dir / "RESULTS.md", summaries, all_final_rows, ", ".join(row["scenario"] for row in scenarios))

    summary = {
        "leakage_policy": {
            "selection_uses_evaluation": False,
            "evaluation_used_for_fixed_evaluation_only": True,
            "uses_official_merlin_split_val": False,
            "selection_source": "train-only metrics",
        },
        "counts": {
            "candidate_configs": len(configs),
            "scenarios": len(scenarios),
            "train_grid_rows": len(all_train_rows),
            "threshold_grid_rows": len(all_grid_rows),
            "train_selected_rows": len(all_selected_rows),
            "final_selected_rows": len(all_final_rows),
            "summary_rows": len(summaries),
            "train_cases": len(inputs["train_case_ids"]),
            "train_presence": dict(Counter(inputs["original_gold"].values())),
            "evaluation_cases": len(final_gold),
            "evaluation_presence": dict(Counter(final_gold.values())),
        },
        "outputs": {
            "plain_rag_train_grid": str(args.output_dir / "plain_rag_train_grid.csv"),
            "plain_rag_threshold_grid": str(args.output_dir / "plain_rag_threshold_grid.csv"),
            "plain_rag_train_selected_configs": str(args.output_dir / "plain_rag_train_selected_configs.csv"),
            "plain_rag_evaluation_fixed_eval": str(args.output_dir / "plain_rag_evaluation_fixed_eval.csv"),
            "plain_rag_baseline_summary": str(args.output_dir / "plain_rag_baseline_summary.csv"),
            "coverage_by_bank": str(args.output_dir / "coverage_by_bank.csv"),
            "results_md": str(args.output_dir / "RESULTS.md"),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"results_md": summary["outputs"]["results_md"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
