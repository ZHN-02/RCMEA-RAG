#!/usr/bin/env python3
"""Focused train-only search for adaptive threshold + ROI veto policies.

This experiment only adds the combined policy family. Configs are selected on
train cases, then applied once to evaluation without retuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


INPUT_ROOT = Path(os.environ.get("RCMEA_INPUT_ROOT", "inputs")).resolve()
OUTPUT_ROOT = Path(os.environ.get("RCMEA_OUTPUT_ROOT", "outputs")).resolve()
OUT_DIR = OUTPUT_ROOT / "step_02_roi_route"
TRAIN_ONLY_RULE_SELECTED_CSV = INPUT_ROOT / "step_01_selected_policies.csv"
EVALUATION_LABELS_CSV = INPUT_ROOT / "evaluation_labels.csv"
REFERENCE_ROI_QUERY_NPZ = INPUT_ROOT / "reference_roi_queries.npz"

from . import step_01_policy_search as comp


RECALL_FLOORS = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--evaluation-labels-csv", type=Path, default=EVALUATION_LABELS_CSV)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--candidate-pool", type=int, default=80)
    return parser.parse_args()


def clean_case_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("case:"):
        text = text.split(":", 1)[1]
    match = re.search(r"AC[0-9a-fA-F]+", text)
    return match.group(0) if match else text


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def final_presence(label: str) -> str:
    if label == "positive_final":
        return "present"
    if label == "negative_final":
        return "absent"
    return ""


def load_final_gold(path: Path) -> dict[str, str]:
    gold: dict[str, str] = {}
    for row in read_csv(path):
        cid = clean_case_id(row.get("case_id"))
        presence = final_presence(str(row.get("final_label") or ""))
        if cid and presence in {"present", "absent"}:
            gold[cid] = presence
    return gold


def scenario_rows() -> list[dict[str, Any]]:
    banks = [
        {"bank_mode": "base_training_bank", "include_extra": "none", "allow_augmented_keys": False},
        {"bank_mode": "augmented_training_bank", "include_extra": "none", "allow_augmented_keys": True},
        {"bank_mode": "augmented_absent_bank", "include_extra": "absent", "allow_augmented_keys": True},
        {"bank_mode": "default_training_bank", "include_extra": "all", "allow_augmented_keys": True},
    ]
    weights = [
        {"weight_mode": "ratio_1to1", "pos_weight": 1.0, "neg_weight": 1.0},
        {"weight_mode": "neg_2x", "pos_weight": 1.0, "neg_weight": 2.0},
        {"weight_mode": "neg_4x", "pos_weight": 1.0, "neg_weight": 4.0},
    ]
    rows: list[dict[str, Any]] = []
    for bank in banks:
        for weight in weights:
            rows.append({**bank, **weight, "scenario": f"{bank['bank_mode']}__{weight['weight_mode']}"})
    return rows


def load_adaptive_seed_configs() -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    seen: set[str] = set()
    if TRAIN_ONLY_RULE_SELECTED_CSV.exists():
        for row in read_csv(TRAIN_ONLY_RULE_SELECTED_CSV):
            if row.get("kind") != "adaptive_threshold":
                continue
            cfg = json.loads(row["config_json"])
            key = comp.config_json(cfg)
            if key in seen:
                continue
            seen.add(key)
            seeds.append(cfg)
    if seeds:
        return seeds
    return [
        {
            "config_id": "SEED_R020844",
            "kind": "adaptive_threshold",
            "preset": "trunk_merlinroi_bal_roi_equal_roi_adaptive",
            "routes": ["fullct", "rsuper", "radgpt", "merlinroi_bal"],
            "weights": {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75, "merlinroi_bal": 1.0},
            "threshold": -4.0,
            "roi_neg_margin": 2,
            "roi_pos_margin": 2,
            "roi_neg_penalty": 3.0,
            "roi_pos_bonus": 1.0,
            "conflict_penalty": 2.0,
        },
        {
            "config_id": "SEED_R021002",
            "kind": "adaptive_threshold",
            "preset": "trunk_merlinroi_bal_roi_equal_roi_adaptive",
            "routes": ["fullct", "rsuper", "radgpt", "merlinroi_bal"],
            "weights": {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75, "merlinroi_bal": 1.0},
            "threshold": 2.0,
            "roi_neg_margin": 2,
            "roi_pos_margin": 2,
            "roi_neg_penalty": 1.0,
            "roi_pos_bonus": 0.0,
            "conflict_penalty": 1.0,
        },
    ]


def focused_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    def add(cfg: dict[str, Any]) -> None:
        row = dict(cfg)
        row["config_id"] = f"ARV{len(configs) + 1:05d}"
        configs.append(row)

    for seed in load_adaptive_seed_configs():
        routes = list(seed["routes"])
        adaptive_sets = {"adaptive_merlin": ["merlinroi_bal"]}
        if all(route in routes for route in comp.ROI_ROUTES):
            adaptive_sets["adaptive_dual"] = comp.ROI_ROUTES
        veto_sets = {"merlin_veto": ["merlinroi_bal"]}
        if all(route in routes for route in comp.ROI_ROUTES):
            veto_sets["dual_veto"] = comp.ROI_ROUTES
        for adaptive_name, adaptive_routes in adaptive_sets.items():
            for veto_name, veto_routes in veto_sets.items():
                for veto_neg_margin in [2, 3, 4, 5]:
                    for veto_present_max in [0, 1]:
                        for top1_absent in [False, True]:
                            for veto_score_max in [-2.0, 0.0, 2.0, 4.0, 6.0]:
                                cfg = dict(seed)
                                cfg.update(
                                    {
                                        "kind": "adaptive_roi_veto",
                                        "preset": f"{seed.get('preset', 'adaptive_seed')}_{adaptive_name}_{veto_name}",
                                        "adaptive_routes": adaptive_routes,
                                        "veto_routes": veto_routes,
                                        "veto_neg_margin": veto_neg_margin,
                                        "veto_present_max": veto_present_max,
                                        "veto_require_top1_absent": top1_absent,
                                        "veto_score_max": veto_score_max,
                                    }
                                )
                                cfg.pop("config_id", None)
                                add(cfg)
    return configs


def build_inputs() -> dict[str, Any]:
    original_gold = comp.load_original_gold(comp.ORIGINAL_TRAIN_JSONL)
    audit_gold_all = comp.load_audit_gold(comp.AUXILIARY_TRAIN_JSONL)
    audit_extra = {cid: label for cid, label in audit_gold_all.items() if cid not in original_gold}
    train_case_ids = sorted(original_gold)
    train_fold_ids = np.asarray([comp.fold_for_case(cid, 5) for cid in train_case_ids], dtype=np.int16)
    raw_routes = {
        "fullct": comp.load_raw_standard_route(comp.FULLCT_ROUTE),
        "rsuper": comp.load_raw_standard_route(comp.RSUPER_ROUTE),
        "radgpt": comp.load_raw_standard_route(comp.RADGPT_ROUTE),
    }
    mer_key_emb, mer_key_rows, mer_meta = comp.load_npz(comp.MERLIN_BAL_KEY_NPZ)
    mer_final_emb, mer_final_rows, mer_final_meta = comp.load_npz(REFERENCE_ROI_QUERY_NPZ)
    original_cases = set(original_gold)
    return {
        "original_gold": original_gold,
        "audit_extra": audit_extra,
        "train_case_ids": train_case_ids,
        "train_fold_ids": train_fold_ids,
        "raw_routes": raw_routes,
        "mer_key": {
            "embeddings": mer_key_emb,
            "rows": mer_key_rows,
            "metadata": mer_meta,
            "query_index_by_case": comp.choose_query_indices(mer_key_rows, original_cases),
        },
        "mer_final": {"embeddings": mer_final_emb, "rows": mer_final_rows, "metadata": mer_final_meta},
    }


def build_final_arrays_for_bank(
    scenario: dict[str, Any],
    final_gold: dict[str, str],
    inputs: dict[str, Any],
    batch_size: int,
    candidate_pool: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    case_ids = sorted(final_gold)
    fold_ids = np.zeros(len(case_ids), dtype=np.int16)
    labels = comp.build_allowed_labels(inputs["original_gold"], inputs["audit_extra"], str(scenario["include_extra"]))
    original_cases = set(inputs["original_gold"])
    extra_cases = set(labels) - original_cases
    allowed_cases = set(labels)
    exclude_cases = set(final_gold)

    stats_by_route: dict[str, dict[str, dict[str, Any]]] = {}
    for route in comp.TRUNK_ROUTES:
        stats_by_route[route] = {
            cid: comp.standard_stats(cid, inputs["raw_routes"][route], labels, allowed_cases, exclude_cases, route)
            for cid in case_ids
        }

    mer_query_index = comp.choose_query_indices(inputs["mer_final"]["rows"], set(case_ids))
    mer_labels = [
        comp.merlin_row_label(row, inputs["original_gold"], {cid: labels[cid] for cid in extra_cases})
        for row in inputs["mer_key"]["rows"]
    ]
    stats_by_route["merlinroi_bal"] = comp.build_embedding_stats_for_queries(
        "merlinroi_bal",
        case_ids,
        mer_query_index,
        inputs["mer_final"]["embeddings"],
        inputs["mer_key"]["embeddings"],
        inputs["mer_key"]["rows"],
        mer_labels,
        allowed_cases,
        exclude_cases,
        allow_augmented_keys=bool(scenario["allow_augmented_keys"]),
        batch_size=batch_size,
        candidate_pool=candidate_pool,
    )

    arr = comp.examples_to_arrays(case_ids, final_gold, fold_ids, stats_by_route)
    coverage = {
        "allowed_key_cases": len(allowed_cases),
        "extra_key_cases": len(extra_cases),
        "final_cases": len(case_ids),
        "route_cases_with_hits": {route: int(np.sum(arr["known"][:, comp.ROUTE_INDEX[route]] > 0)) for route in comp.ROUTES},
        "merlin_final_query_cases": len(mer_query_index),
    }
    return arr, coverage


def selection_key(metric: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    return (
        -float(metric["fp"]),
        float(metric["specificity"]),
        float(metric["precision"]),
        float(metric["f1"]),
        float(metric["recall"]),
        -float(metric["fn"]),
    )


def train_select_configs(
    arr: dict[str, np.ndarray],
    configs: list[dict[str, Any]],
    sample_weight: np.ndarray,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    y = arr["gold_present"]
    pred_matrix = np.empty((len(configs), y.shape[0]), dtype=bool)
    extra_rows: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs):
        pred, extra = comp.predict_config(arr, cfg)
        pred_matrix[idx] = pred
        metric = comp.metric_from_bool(y, pred, sample_weight)
        extra_rows.append(extra)
        for floor in RECALL_FLOORS:
            if metric["recall"] >= floor:
                objective = f"min_fp_recall_ge_{floor:.2f}".replace(".", "p")
                current = selected.get(objective)
                if current is None or selection_key(metric) > selection_key(current["metric"]):
                    selected[objective] = {"cfg": cfg, "metric": metric, "pred": pred, "extra": extra}
        current = selected.get("best_f1")
        if current is None:
            selected["best_f1"] = {"cfg": cfg, "metric": metric, "pred": pred, "extra": extra}
        else:
            key = (
                float(metric["f1"]),
                float(metric["balanced_accuracy"]),
                float(metric["specificity"]),
                -float(metric["fp"]),
            )
            old = current["metric"]
            old_key = (
                float(old["f1"]),
                float(old["balanced_accuracy"]),
                float(old["specificity"]),
                -float(old["fp"]),
            )
            if key > old_key:
                selected["best_f1"] = {"cfg": cfg, "metric": metric, "pred": pred, "extra": extra}
    rows: list[dict[str, Any]] = []
    for objective, item in selected.items():
        cfg = item["cfg"]
        rows.append(
            {
                "selection_scope": "adaptive_roi_veto",
                "objective": objective,
                "config_id": cfg["config_id"],
                "kind": cfg["kind"],
                "preset": cfg["preset"],
                "config_json": comp.config_json(cfg),
                **comp.metric_prefix("train", item["metric"]),
                "train_veto_n": item["extra"].get("veto_n", 0),
                "train_rescue_n": item["extra"].get("rescue_n", 0),
            }
        )
    return rows


def evaluate_selected(
    selected_rows: list[dict[str, Any]],
    final_arr: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selected_rows:
        cfg = json.loads(row["config_json"])
        pred, extra = comp.predict_config(final_arr, cfg)
        metric = comp.metric_from_bool(final_arr["gold_present"], pred)
        rows.append(
            {
                **row,
                **comp.metric_prefix("evaluation", metric),
                "evaluation_veto_n": extra.get("veto_n", 0),
                "evaluation_rescue_n": extra.get("rescue_n", 0),
            }
        )
    return rows


def format_count(value: Any) -> str:
    return str(int(round(float(value))))


def build_markdown(rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]], path: Path) -> None:
    focus = sorted(
        rows,
        key=lambda row: (
            float(row["evaluation_fp"]),
            -float(row["evaluation_f1"]),
            -float(row["evaluation_recall"]),
            float(row["evaluation_fn"]),
        ),
    )
    lines = [
        "# Adaptive Threshold + ROI Veto Focused Search",
        "",
        "Configs are selected on train only. evaluation is fixed one-shot evaluation only.",
        "",
        "## Lowest-FP Fixed Evaluations",
        "",
        "| scenario | objective | preset | train TP/TN/FP/FN | final TP/TN/FP/FN | final F1/spec/recall | config | veto |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for row in focus[:80]:
        lines.append(
            "| {scenario} | {objective} | {preset} | {ttp}/{ttn}/{tfp}/{tfn} | {ftp}/{ftn}/{ffp}/{ffn} | {f1:.4f}/{spec:.4f}/{rec:.4f} | {cfg} | {veto} |".format(
                scenario=row["scenario"],
                objective=row["objective"],
                preset=row["preset"],
                ttp=format_count(row["train_tp"]),
                ttn=format_count(row["train_tn"]),
                tfp=format_count(row["train_fp"]),
                tfn=format_count(row["train_fn"]),
                ftp=format_count(row["evaluation_tp"]),
                ftn=format_count(row["evaluation_tn"]),
                ffp=format_count(row["evaluation_fp"]),
                ffn=format_count(row["evaluation_fn"]),
                f1=float(row["evaluation_f1"]),
                spec=float(row["evaluation_specificity"]),
                rec=float(row["evaluation_recall"]),
                cfg=row["config_id"],
                veto=format_count(row["evaluation_veto_n"]),
            )
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| bank | final cases | allowed keys | fullct | rsuper | radgpt | merlinroi_bal |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in coverage_rows:
        lines.append(
            "| {bank} | {n} | {keys} | {fullct} | {rsuper} | {radgpt} | {mer} |".format(
                bank=row["bank"],
                n=row["final_cases"],
                keys=row["allowed_key_cases"],
                fullct=row["fullct_cases_with_hits"],
                rsuper=row["rsuper_cases_with_hits"],
                radgpt=row["radgpt_cases_with_hits"],
                mer=row["merlinroi_bal_cases_with_hits"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_gold = load_final_gold(args.evaluation_labels_csv)
    inputs = build_inputs()
    scenarios = scenario_rows()
    configs = focused_configs()
    print(
        json.dumps(
            {
                "train_cases": len(inputs["train_case_ids"]),
                "evaluation_cases": len(final_gold),
                "configs": len(configs),
                "scenarios": len(scenarios),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    selected_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    arrays_by_bank: dict[tuple[str, str, bool], tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]] = {}
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
                inputs["mer_key"],
                args.embedding_batch_size,
                args.candidate_pool,
            )
            final_arr, coverage = build_final_arrays_for_bank(
                scenario,
                final_gold,
                inputs,
                args.embedding_batch_size,
                args.candidate_pool,
            )
            arrays_by_bank[bank] = (train_arr, final_arr, coverage)
            coverage_rows.append(
                {
                    "bank": "|".join([str(x) for x in bank]),
                    "final_cases": coverage["final_cases"],
                    "allowed_key_cases": coverage["allowed_key_cases"],
                    "extra_key_cases": coverage["extra_key_cases"],
                    "merlin_final_query_cases": coverage["merlin_final_query_cases"],
                    **{f"{route}_cases_with_hits": coverage["route_cases_with_hits"][route] for route in comp.ROUTES},
                }
            )
        train_arr, final_arr, _coverage = arrays_by_bank[bank]
        print(f"[scenario] {scenario['scenario']}", flush=True)
        sample_weight = comp.sample_weights_for_scenario(train_arr, scenario)
        rows = train_select_configs(train_arr, configs, sample_weight)
        rows = [{**scenario, **row} for row in rows]
        selected_rows.extend(rows)
        final_rows.extend([{**scenario, **row} for row in evaluate_selected(rows, final_arr)])

    write_csv(args.output_dir / "train_selected_configs.csv", selected_rows)
    write_csv(args.output_dir / "evaluation_fixed_eval.csv", final_rows)
    write_csv(args.output_dir / "coverage_by_bank.csv", coverage_rows)
    build_markdown(final_rows, coverage_rows, args.output_dir / "RESULTS.md")
    summary = {
        "leakage_policy": {
            "selection_uses_evaluation": False,
            "evaluation_used_for_fixed_evaluation_only": True,
            "uses_official_merlin_split_val": False,
            "thresholds_selected_on": "training cases only",
        },
        "counts": {
            "train_cases": len(inputs["train_case_ids"]),
            "train_presence": dict(Counter(inputs["original_gold"].values())),
            "audit_extra_presence": dict(Counter(inputs["audit_extra"].values())),
            "evaluation_cases": len(final_gold),
            "evaluation_presence": dict(Counter(final_gold.values())),
            "configs": len(configs),
            "scenarios": len(scenarios),
            "selected_rows": len(selected_rows),
            "final_rows": len(final_rows),
        },
        "outputs": {
            "train_selected_configs": str(args.output_dir / "train_selected_configs.csv"),
            "evaluation_fixed_eval": str(args.output_dir / "evaluation_fixed_eval.csv"),
            "coverage_by_bank": str(args.output_dir / "coverage_by_bank.csv"),
            "results_md": str(args.output_dir / "RESULTS.md"),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": summary["outputs"]["results_md"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
