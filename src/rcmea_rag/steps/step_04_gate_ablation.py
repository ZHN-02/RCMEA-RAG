#!/usr/bin/env python3
"""Leave-one-route-out RCMEA-RAG ablation on the final 1433 test set.

This runner keeps the existing train-only RAG policy-selection protocol, but
replaces the fixed evaluation set with the final evaluation selected policy low-FP test
split. The final test labels are never used for selecting thresholds.
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


INPUT_ROOT = Path(os.environ.get("RCMEA_INPUT_ROOT", "inputs")).resolve()
OUTPUT_ROOT = Path(os.environ.get("RCMEA_OUTPUT_ROOT", "outputs")).resolve()
OUT_DIR = OUTPUT_ROOT / "step_04_gate_ablation"
TABLE_DIR = OUT_DIR / "tables"
EVALUATION_RECORDS_JSONL = INPUT_ROOT / "evaluation_records.jsonl"
EVALUATION_METADATA_JSON = INPUT_ROOT / "evaluation_metadata.json"

from . import step_01_policy_search as comp
from . import step_02_roi_route as focused


SCENARIO = {
    "bank_mode": "default_training_bank",
    "include_extra": "all",
    "allow_augmented_keys": True,
    "weight_mode": "ratio_1to1",
    "pos_weight": 1.0,
    "neg_weight": 1.0,
    "scenario": "default_training_bank__balanced",
}
TRUNK_WEIGHTS = {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75}
PRIMARY_OBJECTIVE = "min_fp_recall_ge_0p850"
RECALL_FLOORS = [0.90, 0.875, 0.85, 0.825, 0.80, 0.775, 0.75]
SPEC_FLOORS = [0.85, 0.825, 0.80, 0.775, 0.75]
FP_RATE_BUDGETS = [0.12, 0.15, 0.175, 0.20, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--table-path", type=Path, default=TABLE_DIR / "step_04_ablation_table.tex")
    parser.add_argument("--evaluation-records-jsonl", type=Path, default=EVALUATION_RECORDS_JSONL)
    parser.add_argument("--evaluation-metadata-json", type=Path, default=EVALUATION_METADATA_JSON)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--quick", action="store_true", help="Use a smaller grid for smoke testing.")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def cfg_json(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def metric_prefix(prefix: str, metric: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metric.items()}


def count_str(row: dict[str, Any], prefix: str) -> str:
    return "/".join(str(int(round(float(row[f"{prefix}_{key}"])))) for key in ["tp", "tn", "fp", "fn"])


def load_gold_from_jsonl(path: Path) -> dict[str, str]:
    gold: dict[str, str] = {}
    for row in iter_jsonl(path):
        cid = comp.clean_case_id(row.get("case_id"))
        target_raw = row.get("target")
        target = json.loads(target_raw) if isinstance(target_raw, str) else target_raw
        presence = str((target or {}).get("lesion_presence") or "").strip().lower()
        if cid and presence in {"present", "absent"}:
            gold[cid] = presence
    return gold


def load_locked_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload["config"]
    if cfg.get("config_id") != "selected policy":
        raise RuntimeError(f"Unexpected locked config: {cfg.get('config_id')}")
    return cfg


def route_idx(routes: list[str]) -> list[int]:
    return [comp.ROUTE_INDEX[route] for route in routes]


def weighted_score(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    routes = list(cfg.get("routes") or [])
    if not routes:
        return np.zeros(arr["gold_present"].shape[0], dtype=np.float32)
    idx = route_idx(routes)
    weights = np.asarray([float((cfg.get("weights") or {}).get(route, 0.0)) for route in comp.ROUTES], dtype=np.float32)
    return arr["margins"][:, idx] @ weights[idx]


def roi_margins(arr: dict[str, np.ndarray], routes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not routes:
        n = arr["gold_present"].shape[0]
        empty_float = np.zeros((n, 0), dtype=np.float32)
        empty_bool = np.zeros((n, 0), dtype=bool)
        return empty_float, empty_float, empty_bool, empty_bool
    idx = route_idx(routes)
    neg = arr["absent_n"][:, idx] - arr["present_n"][:, idx]
    pos = arr["present_n"][:, idx] - arr["absent_n"][:, idx]
    top1_abs = arr["top1_absent"][:, idx]
    top1_pos = arr["top1_present"][:, idx]
    return neg, pos, top1_abs, top1_pos


def trunk_score(arr: dict[str, np.ndarray], routes: list[str] | None = None) -> np.ndarray:
    score = np.zeros(arr["gold_present"].shape[0], dtype=np.float32)
    for route in routes or comp.TRUNK_ROUTES:
        score += float(TRUNK_WEIGHTS[route]) * arr["margins"][:, comp.ROUTE_INDEX[route]]
    return score


def counter_mask(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    routes = list(cfg.get("counter_routes") or [])
    if not routes:
        return np.zeros(arr["gold_present"].shape[0], dtype=bool)
    idx = route_idx(routes)
    neg, _pos, top1_abs, _top1_pos = roi_margins(arr, routes)
    mask = (arr["known"][:, idx] > 0) & (neg >= int(cfg.get("neg_margin", 2)))
    mask &= arr["present_n"][:, idx] <= int(cfg.get("present_max", 1))
    if bool(cfg.get("require_top1_absent")):
        mask &= top1_abs
    return np.any(mask, axis=1)


def dual_counter_mask(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    routes = list(cfg.get("counter_routes") or [])
    if not routes:
        return np.zeros(arr["gold_present"].shape[0], dtype=bool)
    idx = route_idx(routes)
    neg, _pos, top1_abs, _top1_pos = roi_margins(arr, routes)
    mask = (arr["known"][:, idx] > 0) & (neg >= int(cfg.get("neg_margin", 2)))
    mask &= arr["present_n"][:, idx] <= int(cfg.get("present_max", 1))
    if bool(cfg.get("require_top1_absent")):
        mask &= top1_abs
    return np.sum(mask, axis=1) >= int(cfg.get("counter_route_min", 2))


def positive_roi_mask(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    routes = list(cfg.get("rescue_routes") or cfg.get("counter_routes") or [])
    if not routes:
        return np.zeros(arr["gold_present"].shape[0], dtype=bool)
    idx = route_idx(routes)
    _neg, pos, _top1_abs, top1_pos = roi_margins(arr, routes)
    mask = (arr["known"][:, idx] > 0) & (pos >= int(cfg.get("pos_margin", 2)))
    mask &= arr["absent_n"][:, idx] <= int(cfg.get("absent_max", 1))
    if bool(cfg.get("require_top1_present")):
        mask &= top1_pos
    return np.sum(mask, axis=1) >= int(cfg.get("rescue_route_min", 1))


def max_counter_strength(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    routes = list(cfg.get("counter_routes") or [])
    if not routes:
        return np.zeros(arr["gold_present"].shape[0], dtype=np.float32)
    neg, _pos, _top1_abs, _top1_pos = roi_margins(arr, routes)
    strength = np.maximum(0.0, np.max(neg, axis=1).astype(np.float32) - float(cfg.get("neg_margin", 2)) + 1.0)
    return np.clip(strength, 0.0, 6.0)


def max_positive_strength(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    routes = list(cfg.get("rescue_routes") or cfg.get("counter_routes") or [])
    if not routes:
        return np.zeros(arr["gold_present"].shape[0], dtype=np.float32)
    _neg, pos, _top1_abs, _top1_pos = roi_margins(arr, routes)
    strength = np.maximum(0.0, np.max(pos, axis=1).astype(np.float32) - float(cfg.get("pos_margin", 2)) + 1.0)
    return np.clip(strength, 0.0, 6.0)


def predict_config(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    kind = cfg["kind"]
    if kind == "weighted_threshold":
        score = weighted_score(arr, cfg)
        pred = score > float(cfg["threshold"])
        return pred, {
            "score_mean": float(np.mean(score)),
            "base_present_n": int(np.sum(pred)),
            "veto_n": 0,
            "counter_n": 0,
            "dual_counter_n": 0,
            "roi_positive_n": 0,
            "conflict_n": 0,
            "risk_mean": 0.0,
        }
    if kind == "risk_adjusted_margin":
        score = weighted_score(arr, cfg)
        ctr_strength = max_counter_strength(arr, cfg)
        pos_strength = max_positive_strength(arr, cfg)
        ctr = ctr_strength > 0
        dual = dual_counter_mask(arr, cfg)
        conflict = (trunk_score(arr, list(cfg.get("routes") or [])) > float(cfg.get("conflict_trunk_min", 0.0))) & ctr
        threshold = (
            float(cfg["threshold"])
            + float(cfg.get("neg_penalty", 0.0)) * ctr_strength
            + float(cfg.get("conflict_penalty", 0.0)) * conflict.astype(np.float32)
            + float(cfg.get("dual_penalty", 0.0)) * dual.astype(np.float32)
            - float(cfg.get("pos_bonus", 0.0)) * pos_strength
        )
        pred = score > threshold
        return pred, {
            "score_mean": float(np.mean(score)),
            "base_present_n": int(np.sum(pred)),
            "veto_n": 0,
            "counter_n": int(np.sum(counter_mask(arr, cfg))),
            "dual_counter_n": int(np.sum(dual)),
            "roi_positive_n": int(np.sum(positive_roi_mask(arr, cfg))),
            "conflict_n": int(np.sum(conflict)),
            "risk_mean": float(np.mean(ctr_strength)),
        }
    if kind == "posthoc_risk_veto":
        base_cfg = dict(cfg["base_cfg"])
        base_pred, base_extra = predict_config(arr, base_cfg)
        score = weighted_score(arr, base_cfg)
        ctr_strength = max_counter_strength(arr, base_cfg)
        pos_strength = max_positive_strength(arr, base_cfg)
        ctr = counter_mask(arr, base_cfg)
        dual = dual_counter_mask(arr, base_cfg)
        conflict = (trunk_score(arr, list(base_cfg.get("routes") or [])) > float(base_cfg.get("conflict_trunk_min", 0.0))) & ctr
        risk = (
            float(cfg.get("counter_alpha", 1.0)) * ctr_strength
            + float(cfg.get("conflict_weight", 0.0)) * conflict.astype(np.float32)
            + float(cfg.get("dual_weight", 0.0)) * dual.astype(np.float32)
            - float(cfg.get("pos_credit", 0.0)) * pos_strength
        )
        veto = base_pred & (risk >= float(cfg["risk_threshold"])) & (score <= float(cfg["score_max"]))
        pred = base_pred & ~veto
        return pred, {
            "score_mean": float(np.mean(score)),
            "base_present_n": int(base_extra.get("base_present_n", int(np.sum(base_pred)))),
            "veto_n": int(np.sum(veto)),
            "counter_n": int(np.sum(ctr)),
            "dual_counter_n": int(np.sum(dual)),
            "roi_positive_n": int(np.sum(positive_roi_mask(arr, base_cfg))),
            "conflict_n": int(np.sum(conflict)),
            "risk_mean": float(np.mean(risk)),
        }
    raise ValueError(kind)


def route_variants() -> list[dict[str, Any]]:
    return [
        {
            "variant": "full_rcmea_rag",
            "changed_component": "none",
            "mechanism": "fullct+rsuper+radgpt trunk with Merlin ROI counter-veto",
            "routes": ["fullct", "rsuper", "radgpt"],
            "counter_routes": ["merlinroi_bal"],
            "rescue_routes": ["merlinroi_bal"],
        },
        {
            "variant": "wo_fullct",
            "changed_component": "remove full-CT image retrieval",
            "mechanism": "tests global CT visual route",
            "routes": ["rsuper", "radgpt"],
            "counter_routes": ["merlinroi_bal"],
            "rescue_routes": ["merlinroi_bal"],
        },
        {
            "variant": "wo_rsuper",
            "changed_component": "remove clinical-neighbor retrieval",
            "mechanism": "tests clinical text-neighbor route",
            "routes": ["fullct", "radgpt"],
            "counter_routes": ["merlinroi_bal"],
            "rescue_routes": ["merlinroi_bal"],
        },
        {
            "variant": "wo_radgpt",
            "changed_component": "remove structured-schema retrieval",
            "mechanism": "tests structured schema route",
            "routes": ["fullct", "rsuper"],
            "counter_routes": ["merlinroi_bal"],
            "rescue_routes": ["merlinroi_bal"],
        },
        {
            "variant": "wo_merlinroi_branch",
            "changed_component": "remove Merlin ROI counter branch",
            "mechanism": "tests ROI risk counter-evidence",
            "routes": ["fullct", "rsuper", "radgpt"],
            "counter_routes": [],
            "rescue_routes": [],
        },
    ]


def weights_for(routes: list[str]) -> dict[str, float]:
    return {route: TRUNK_WEIGHTS[route] for route in routes}


def make_risk_cfgs(spec: dict[str, Any], quick: bool) -> list[dict[str, Any]]:
    if not spec["counter_routes"]:
        thresholds = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 1.0, 2.0] if not quick else [-2.0, -1.0, 0.0]
        return [
            {
                "kind": "weighted_threshold",
                "family": "weighted_trunk_no_roi",
                "preset": spec["variant"],
                "variant": spec["variant"],
                "routes": spec["routes"],
                "weights": weights_for(spec["routes"]),
                "threshold": threshold,
            }
            for threshold in thresholds
        ]

    thresholds = [-1.5, -1.0, -0.5, 0.0, 0.5] if not quick else [-1.5, -1.0, 0.0]
    neg_margins = [2, 3, 4] if not quick else [2, 3]
    present_max_values = [0, 1]
    neg_penalties = [0.5, 1.0]
    conflict_penalties = [0.0, 2.0, 4.0]
    dual_penalties = [0.0, 1.0, 2.0] if not quick else [0.0, 2.0]
    pos_bonuses = [0.0, 1.0]
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for neg_margin in neg_margins:
            for present_max in present_max_values:
                for neg_penalty in neg_penalties:
                    for conflict_penalty in conflict_penalties:
                        for dual_penalty in dual_penalties:
                            for pos_bonus in pos_bonuses:
                                rows.append(
                                    {
                                        "kind": "risk_adjusted_margin",
                                        "family": "leave_one_route_risk_adjusted",
                                        "preset": spec["variant"],
                                        "variant": spec["variant"],
                                        "routes": spec["routes"],
                                        "weights": weights_for(spec["routes"]),
                                        "counter_routes": spec["counter_routes"],
                                        "rescue_routes": spec["rescue_routes"],
                                        "threshold": threshold,
                                        "neg_margin": neg_margin,
                                        "present_max": present_max,
                                        "pos_margin": 2,
                                        "absent_max": 1,
                                        "counter_route_min": 2,
                                        "neg_penalty": neg_penalty,
                                        "conflict_penalty": conflict_penalty,
                                        "dual_penalty": dual_penalty,
                                        "pos_bonus": pos_bonus,
                                        "conflict_trunk_min": 0.0,
                                    }
                                )
    return rows


def make_posthoc_cfgs(spec: dict[str, Any], quick: bool) -> list[dict[str, Any]]:
    if not spec["counter_routes"]:
        return []
    base_cfgs = make_risk_cfgs(spec, quick)
    if quick:
        base_cfgs = [
            cfg
            for cfg in base_cfgs
            if cfg["threshold"] in {-1.0, 0.0}
            and cfg["neg_margin"] in {2, 3}
            and cfg["present_max"] == 1
            and cfg["neg_penalty"] in {0.5, 1.0}
            and cfg["conflict_penalty"] in {0.0, 2.0}
            and cfg["pos_bonus"] in {0.0}
        ]
    else:
        base_cfgs = [
            cfg
            for cfg in base_cfgs
            if cfg["threshold"] in {-1.5, -1.0, 0.0}
            and cfg["neg_margin"] in {2, 3}
            and cfg["present_max"] == 1
            and cfg["neg_penalty"] in {0.5, 1.0}
            and cfg["conflict_penalty"] in {0.0, 2.0, 4.0}
            and cfg["dual_penalty"] in {0.0, 2.0}
            and cfg["pos_bonus"] in {0.0, 1.0}
        ]
    risk_thresholds = [1.0, 2.0, 3.0] if not quick else [1.0, 3.0]
    score_max_values = [4.0, 6.0]
    conflict_weights = [0.0, 1.0] if not quick else [0.0, 1.0]
    dual_weights = [0.0]
    pos_credits = [0.0]
    rows: list[dict[str, Any]] = []
    for base_cfg in base_cfgs:
        for risk_threshold in risk_thresholds:
            for score_max in score_max_values:
                for conflict_weight in conflict_weights:
                    for dual_weight in dual_weights:
                        for pos_credit in pos_credits:
                            rows.append(
                                {
                                    "kind": "posthoc_risk_veto",
                                    "family": "leave_one_route_posthoc_veto",
                                    "preset": f"{spec['variant']}_posthoc_veto",
                                    "variant": spec["variant"],
                                    "base_cfg": base_cfg,
                                    "risk_threshold": risk_threshold,
                                    "score_max": score_max,
                                    "counter_alpha": 1.0,
                                    "conflict_weight": conflict_weight,
                                    "dual_weight": dual_weight,
                                    "pos_credit": pos_credit,
                                }
                            )
    return rows


def candidate_configs(quick: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spec_by_variant = {spec["variant"]: spec for spec in route_variants()}
    for spec in route_variants():
        rows.extend(make_risk_cfgs(spec, quick))
        rows.extend(make_posthoc_cfgs(spec, quick))
    for idx, row in enumerate(rows, start=1):
        row["config_id"] = f"LOR{idx:06d}"
        spec = spec_by_variant[row["variant"]]
        row["changed_component"] = spec["changed_component"]
        row["mechanism"] = spec["mechanism"]
    return rows


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
    key: tuple[str, str],
    cfg: dict[str, Any],
    metric: dict[str, float],
    extra: dict[str, Any],
    mode: str,
) -> None:
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


def select_on_train(
    arr: dict[str, np.ndarray],
    configs: list[dict[str, Any]],
    sample_weight: np.ndarray,
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    y = arr["gold_present"]
    for cfg in configs:
        pred, extra = predict_config(arr, cfg)
        metric = comp.metric_from_bool(y, pred, sample_weight)
        neg_total = float(metric["tn"] + metric["fp"])
        fp_rate = float(metric["fp"]) / neg_total if neg_total else 0.0
        scopes = [cfg["variant"], f"{cfg['variant']}::{cfg['kind']}", "global"]
        for scope in scopes:
            update_selected(selected, (scope, "best_f1"), cfg, metric, extra, "best_f1")
            for recall in RECALL_FLOORS:
                if float(metric["recall"]) >= recall:
                    update_selected(selected, (scope, f"min_fp_recall_ge_{recall:.3f}".replace(".", "p")), cfg, metric, extra, "min_fp")
            for spec in SPEC_FLOORS:
                if float(metric["specificity"]) >= spec:
                    update_selected(selected, (scope, f"max_recall_spec_ge_{spec:.3f}".replace(".", "p")), cfg, metric, extra, "max_recall")
            for budget in FP_RATE_BUDGETS:
                if fp_rate <= budget:
                    update_selected(selected, (scope, f"max_recall_fpr_le_{budget:.3f}".replace(".", "p")), cfg, metric, extra, "max_recall")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for (scope, objective), item in selected.items():
        cfg = item["cfg"]
        unique = (scope, objective, cfg_json(cfg))
        if unique in seen:
            continue
        seen.add(unique)
        metric = item["metric"]
        neg_total = float(metric["tn"] + metric["fp"])
        rows.append(
            {
                "selection_scope": scope,
                "objective": objective,
                "variant": cfg["variant"],
                "changed_component": cfg["changed_component"],
                "mechanism": cfg["mechanism"],
                "config_id": cfg["config_id"],
                "kind": cfg["kind"],
                "family": cfg["family"],
                "preset": cfg["preset"],
                "routes": "+".join(cfg.get("routes") or cfg.get("base_cfg", {}).get("routes") or []),
                "counter_routes": "+".join(cfg.get("counter_routes") or cfg.get("base_cfg", {}).get("counter_routes") or []),
                "config_json": cfg_json(cfg),
                "train_fpr": float(metric["fp"]) / neg_total if neg_total else 0.0,
                **metric_prefix("train", metric),
                **{f"train_{key}": value for key, value in item["extra"].items()},
            }
        )
    return rows


def evaluate_rows(rows: list[dict[str, Any]], arr: dict[str, np.ndarray], prefix: str = "evaluation") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        unique = (row["selection_scope"], row["objective"], row["config_json"])
        if unique in seen:
            continue
        seen.add(unique)
        cfg = json.loads(row["config_json"])
        pred, extra = predict_config(arr, cfg)
        metric = comp.metric_from_bool(arr["gold_present"], pred)
        out.append({**row, **metric_prefix(prefix, metric), **{f"{prefix}_{key}": value for key, value in extra.items()}})
    return out


def locked_row(locked_cfg: dict[str, Any], train_arr: dict[str, np.ndarray], final_arr: dict[str, np.ndarray], sample_weight: np.ndarray) -> dict[str, Any]:
    cfg = dict(locked_cfg)
    cfg["variant"] = "locked_policy"
    cfg["changed_component"] = "none"
    cfg["mechanism"] = "locked policy"
    train_pred, train_extra = predict_config(train_arr, cfg)
    final_pred, final_extra = predict_config(final_arr, cfg)
    train_metric = comp.metric_from_bool(train_arr["gold_present"], train_pred, sample_weight)
    final_metric = comp.metric_from_bool(final_arr["gold_present"], final_pred)
    neg_total = float(train_metric["tn"] + train_metric["fp"])
    return {
        "selection_scope": "locked_policy",
        "objective": PRIMARY_OBJECTIVE,
        "variant": "locked_policy",
        "changed_component": "none",
        "mechanism": "locked policy",
        "config_id": cfg["config_id"],
        "kind": cfg["kind"],
        "family": cfg["family"],
        "preset": cfg["preset"],
        "routes": "+".join(cfg["base_cfg"]["routes"]),
        "counter_routes": "+".join(cfg["base_cfg"].get("counter_routes") or []),
        "config_json": cfg_json(cfg),
        "train_fpr": float(train_metric["fp"]) / neg_total if neg_total else 0.0,
        **metric_prefix("train", train_metric),
        **{f"train_{key}": value for key, value in train_extra.items()},
        **metric_prefix("evaluation", final_metric),
        **{f"evaluation_{key}": value for key, value in final_extra.items()},
    }


def fixed_knockout_rows(
    locked_cfg: dict[str, Any],
    train_arr: dict[str, np.ndarray],
    final_arr: dict[str, np.ndarray],
    sample_weight: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for drop_route in ["fullct", "rsuper", "radgpt", "merlinroi_bal"]:
        cfg = json.loads(cfg_json(locked_cfg))
        cfg["variant"] = f"locked_drop_{drop_route}"
        cfg["changed_component"] = f"fixed-policy knockout of {drop_route}"
        cfg["mechanism"] = "same selected policy hyperparameters after removing one route"
        base = cfg["base_cfg"]
        if drop_route in base["routes"]:
            base["routes"] = [route for route in base["routes"] if route != drop_route]
            base["weights"] = {route: value for route, value in base["weights"].items() if route != drop_route}
        if drop_route in base.get("counter_routes", []):
            base["counter_routes"] = [route for route in base["counter_routes"] if route != drop_route]
            base["rescue_routes"] = [route for route in base.get("rescue_routes", []) if route != drop_route]
        cfg["config_id"] = f"selected policy_drop_{drop_route}"
        train_pred, train_extra = predict_config(train_arr, cfg)
        final_pred, final_extra = predict_config(final_arr, cfg)
        train_metric = comp.metric_from_bool(train_arr["gold_present"], train_pred, sample_weight)
        final_metric = comp.metric_from_bool(final_arr["gold_present"], final_pred)
        neg_total = float(train_metric["tn"] + train_metric["fp"])
        rows.append(
            {
                "selection_scope": "locked_fixed_knockout",
                "objective": "fixed_policy_no_reselection",
                "variant": cfg["variant"],
                "changed_component": cfg["changed_component"],
                "mechanism": cfg["mechanism"],
                "config_id": cfg["config_id"],
                "kind": cfg["kind"],
                "family": cfg["family"],
                "preset": cfg["preset"],
                "routes": "+".join(base.get("routes") or []),
                "counter_routes": "+".join(base.get("counter_routes") or []),
                "config_json": cfg_json(cfg),
                "train_fpr": float(train_metric["fp"]) / neg_total if neg_total else 0.0,
                **metric_prefix("train", train_metric),
                **{f"train_{key}": value for key, value in train_extra.items()},
                **metric_prefix("evaluation", final_metric),
                **{f"evaluation_{key}": value for key, value in final_extra.items()},
            }
        )
    return rows


def choose_primary_rows(final_rows: list[dict[str, Any]], locked: dict[str, Any]) -> list[dict[str, Any]]:
    primary: list[dict[str, Any]] = [locked]
    for spec in route_variants():
        if spec["variant"] == "full_rcmea_rag":
            continue
        candidates = [
            row
            for row in final_rows
            if row["selection_scope"] == spec["variant"] and row["objective"] == PRIMARY_OBJECTIVE
        ]
        if not candidates:
            candidates = [row for row in final_rows if row["selection_scope"] == spec["variant"] and row["objective"] == "best_f1"]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: (float(row["evaluation_fp"]), -float(row["evaluation_f1"]), float(row["evaluation_fn"])))
        primary.append(best)
    return primary


def build_markdown(
    path: Path,
    primary_rows: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# RCMEA-RAG Leave-One-Route-Out On Evaluation",
        "",
        f"Final test: `{args.evaluation_records_jsonl}`",
        f"Locked policy source: `{args.evaluation_metadata_json}`",
        "",
        "Policy/config selection uses training metrics only; the 1433-case final test is fixed one-shot evaluation.",
        "",
        "## Primary Train-Selected Route Ablation",
        "",
        "| variant | changed component | objective | final TP/TN/FP/FN | F1 | Spec | Recall | train TP/TN/FP/FN | config |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in primary_rows:
        lines.append(
            "| {variant} | {changed} | {objective} | {counts} | {f1:.4f} | {spec:.4f} | {rec:.4f} | {train_counts} | {cfg} |".format(
                variant=row["variant"],
                changed=row["changed_component"],
                objective=row["objective"],
                counts=count_str(row, "evaluation"),
                f1=float(row["evaluation_f1"]),
                spec=float(row["evaluation_specificity"]),
                rec=float(row["evaluation_recall"]),
                train_counts=count_str(row, "train"),
                cfg=row["config_id"],
            )
        )
    lines.extend(
        [
            "",
            "## Fixed-Policy Knockout Diagnostic",
            "",
            "| variant | final TP/TN/FP/FN | F1 | Spec | Recall | train TP/TN/FP/FN |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in fixed_rows:
        lines.append(
            "| {variant} | {counts} | {f1:.4f} | {spec:.4f} | {rec:.4f} | {train_counts} |".format(
                variant=row["variant"],
                counts=count_str(row, "evaluation"),
                f1=float(row["evaluation_f1"]),
                spec=float(row["evaluation_specificity"]),
                rec=float(row["evaluation_recall"]),
                train_counts=count_str(row, "train"),
            )
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| final cases | fullct | rsuper | radgpt | merlinroi_bal | allowed keys | extra keys |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| {final_cases} | {fullct} | {rsuper} | {radgpt} | {merlinroi_bal} | {allowed} | {extra} |".format(
                final_cases=coverage["final_cases"],
                fullct=coverage["route_cases_with_hits"]["fullct"],
                rsuper=coverage["route_cases_with_hits"]["rsuper"],
                radgpt=coverage["route_cases_with_hits"]["radgpt"],
                merlinroi_bal=coverage["route_cases_with_hits"]["merlinroi_bal"],
                allowed=coverage["allowed_key_cases"],
                extra=coverage["extra_key_cases"],
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(text: Any) -> str:
    return str(text).replace("_", "\\_").replace(">=", "$\\geq$").replace("<=", "$\\leq$")


def build_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = {
        "locked_policy": "Locked RCMEA-RAG",
        "wo_fullct": "w/o full CT",
        "wo_rsuper": "w/o clinical neighbor",
        "wo_radgpt": "w/o structured schema",
        "wo_merlinroi_branch": "w/o Merlin ROI branch",
    }
    changed_labels = {
        "locked_policy": "None",
        "wo_fullct": "Full-CT route",
        "wo_rsuper": "Clinical-neighbor route",
        "wo_radgpt": "Structured-schema route",
        "wo_merlinroi_branch": "Merlin ROI counter branch",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Leave-one-route-out ablation of RCMEA-RAG on the final 1433-case test set. Each ablated family selects its operating point on training data only; the final test is used once for fixed evaluation.}",
        "\\label{tab:rcmea_rag_leave_one_route_evaluation}",
        "\\small",
        "\\begin{tabular}{l l c c c c}",
        "\\toprule",
        "Variant & Changed component & TP/TN/FP/FN & F1 & Spec. & Rec. \\\\",
        "\\midrule",
    ]
    for row in rows:
        variant = labels.get(row["variant"], row["variant"])
        lines.append(
            "{variant} & {changed} & {counts} & {f1:.4f} & {spec:.4f} & {rec:.4f} \\\\".format(
                variant=latex_escape(variant),
                changed=latex_escape(changed_labels.get(row["variant"], row["changed_component"])),
                counts=count_str(row, "evaluation"),
                f1=float(row["evaluation_f1"]),
                spec=float(row["evaluation_specificity"]),
                rec=float(row["evaluation_recall"]),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_gold = load_gold_from_jsonl(args.evaluation_records_jsonl)
    locked_cfg = load_locked_config(args.evaluation_metadata_json)
    inputs = focused.build_inputs()
    train_arr, train_coverage = comp.build_arrays_for_fold(
        SCENARIO,
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
    final_arr, final_coverage = focused.build_final_arrays_for_bank(
        SCENARIO,
        final_gold,
        inputs,
        args.embedding_batch_size,
        args.candidate_pool,
    )
    sample_weight = comp.sample_weights_for_scenario(train_arr, SCENARIO)
    configs = candidate_configs(args.quick)
    print(
        json.dumps(
            {
                "evaluation_cases": len(final_gold),
                "evaluation_presence": dict(Counter(final_gold.values())),
                "train_cases": len(inputs["train_case_ids"]),
                "configs": len(configs),
                "quick": bool(args.quick),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    selected = select_on_train(train_arr, configs, sample_weight)
    final_rows = evaluate_rows(selected, final_arr)
    locked = locked_row(locked_cfg, train_arr, final_arr, sample_weight)
    fixed_rows = fixed_knockout_rows(locked_cfg, train_arr, final_arr, sample_weight)
    primary_rows = choose_primary_rows(final_rows, locked)

    write_csv(args.output_dir / "leave_one_route_out_train_selected.csv", selected)
    write_csv(args.output_dir / "leave_one_route_out_evaluation_fixed_eval.csv", final_rows)
    write_csv(args.output_dir / "leave_one_route_out_primary_summary.csv", primary_rows)
    write_csv(args.output_dir / "locked_policy_fixed_knockout_evaluation.csv", fixed_rows)
    write_csv(
        args.output_dir / "coverage_by_bank.csv",
        [
            {
                "split": "train",
                "cases": len(inputs["train_case_ids"]),
                "allowed_key_cases": train_coverage["allowed_key_cases"],
                "extra_key_cases": train_coverage["extra_key_cases"],
                **{f"{route}_cases_with_hits": train_coverage["route_cases_with_hits"][route] for route in comp.ROUTES},
            },
            {
                "split": "evaluation",
                "cases": len(final_gold),
                "allowed_key_cases": final_coverage["allowed_key_cases"],
                "extra_key_cases": final_coverage["extra_key_cases"],
                **{f"{route}_cases_with_hits": final_coverage["route_cases_with_hits"][route] for route in comp.ROUTES},
            },
        ],
    )
    build_markdown(args.output_dir / "RESULTS.md", primary_rows, fixed_rows, final_coverage, args)
    build_latex_table(args.table_path, primary_rows)
    summary = {
        "leakage_policy": {
            "selection_uses_evaluation": False,
            "evaluation_used_for_fixed_evaluation_only": True,
            "selection_source": "train-only metrics from existing RCMEA-RAG protocol",
            "locked_policy_source": str(args.evaluation_metadata_json),
            "final_test_source": str(args.evaluation_records_jsonl),
        },
        "scenario": SCENARIO,
        "counts": {
            "candidate_configs": len(configs),
            "train_selected_rows": len(selected),
            "final_rows": len(final_rows),
            "primary_rows": len(primary_rows),
            "fixed_knockout_rows": len(fixed_rows),
            "train_cases": len(inputs["train_case_ids"]),
            "train_presence": dict(Counter(inputs["original_gold"].values())),
            "evaluation_cases": len(final_gold),
            "evaluation_presence": dict(Counter(final_gold.values())),
        },
        "outputs": {
            "train_selected": str(args.output_dir / "leave_one_route_out_train_selected.csv"),
            "evaluation_fixed_eval": str(args.output_dir / "leave_one_route_out_evaluation_fixed_eval.csv"),
            "primary_summary": str(args.output_dir / "leave_one_route_out_primary_summary.csv"),
            "fixed_knockout": str(args.output_dir / "locked_policy_fixed_knockout_evaluation.csv"),
            "coverage": str(args.output_dir / "coverage_by_bank.csv"),
            "results_md": str(args.output_dir / "RESULTS.md"),
            "latex_table": str(args.table_path),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"results_md": summary["outputs"]["results_md"], "latex_table": summary["outputs"]["latex_table"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
