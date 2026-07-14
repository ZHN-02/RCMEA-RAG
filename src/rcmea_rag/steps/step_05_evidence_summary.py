#!/usr/bin/env python3
"""Build evaluation evidence tables for the RCMEA-RAG paper draft.

The script reuses the locked evaluation runner and the plain-RAG baseline code.
All operating points are selected on the train split or are fixed
perturbations of the locked selected policy policy. The evaluation split is used
only for fixed evaluation and analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


INPUT_ROOT = Path(os.environ.get("RCMEA_INPUT_ROOT", "inputs")).resolve()
OUTPUT_ROOT = Path(os.environ.get("RCMEA_OUTPUT_ROOT", "outputs")).resolve()
OUT_DIR = OUTPUT_ROOT / "step_05_evidence_summary"
TABLE_DIR = OUT_DIR / "tables"

EVALUATION_RECORDS_JSONL = INPUT_ROOT / "evaluation_records.jsonl"
EVALUATION_METADATA_JSON = INPUT_ROOT / "evaluation_metadata.json"
AUXILIARY_INPUT_DIR = INPUT_ROOT / "auxiliary"
CANONICAL_INPUT_DIR = INPUT_ROOT / "canonical"
METRIC_COMPONENTS_CSV = INPUT_ROOT / "metric_components.csv"

from . import step_04_gate_ablation as lor
from . import step_03_rag_baselines as plain
from . import step_02_roi_route as focused
from . import step_01_policy_search as comp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--evaluation-records-jsonl", type=Path, default=EVALUATION_RECORDS_JSONL)
    parser.add_argument("--evaluation-metadata-json", type=Path, default=EVALUATION_METADATA_JSON)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--candidate-pool", type=int, default=80)
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric_prefix(prefix: str, metric: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metric.items()}


def count_str(row: dict[str, Any], prefix: str) -> str:
    return "/".join(str(int(round(float(row[f"{prefix}_{key}"])))) for key in ["tp", "tn", "fp", "fn"])


def latex_escape(text: Any) -> str:
    value = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in value)


def fmt(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def load_case_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        cid = comp.clean_case_id(row.get("case_id"))
        target_raw = row.get("target")
        target = json.loads(target_raw) if isinstance(target_raw, str) else target_raw
        target = target or {}
        rows[cid] = {
            "case_id": cid,
            "target_presence": str(target.get("lesion_presence") or "").lower(),
            "target_major": str(target.get("lesion_type_major") or ""),
            "target_detail": str(target.get("lesion_type_detail") or ""),
            "target_locations": "|".join(target.get("lesion_locations") or []),
            "target_modifiers": "|".join(target.get("modifiers") or []),
            "target_report_section": str(target.get("report_section") or ""),
        }
    return rows


def load_canonical_map(path: Path) -> dict[str, dict[str, str]]:
    return {comp.clean_case_id(row.get("case_id")): row for row in read_csv(path)}


def category_for_case(meta: dict[str, Any], canonical: dict[str, str] | None) -> str:
    text_parts = [
        meta.get("target_major", ""),
        meta.get("target_detail", ""),
        meta.get("target_locations", ""),
        meta.get("target_modifiers", ""),
        meta.get("target_report_section", ""),
    ]
    if canonical:
        text_parts.extend(
            [
                canonical.get("rules_locked_category", ""),
                canonical.get("final_mutual_category", ""),
                canonical.get("expanded_positive_categories", ""),
                canonical.get("pancreas_section", ""),
            ]
        )
    text = " ".join(str(part).lower() for part in text_parts)
    if re.search(r"ipmn|cyst|cystic|serous|mucinous", text):
        return "cyst_or_ipmn"
    if re.search(r"mass|tumou?r|neoplasm|cancer|adenocarcinoma|soft.tissue", text):
        return "mass_or_neoplasm"
    if re.search(r"duct|mpd|dilat|dilatation|dilation|prominen", text):
        return "duct_abnormality"
    if re.search(r"hypodense|hypoattenuat|low.attenuation|focal", text):
        return "focal_attenuation_lesion"
    if re.search(r"pancreatitis|inflamm|stranding|edema", text):
        return "inflammation_or_pancreatitis"
    if re.search(r"fluid.collection|collection|necrosis|walled.off|pseudocyst|abscess", text):
        return "fluid_collection_or_necrosis"
    if re.search(r"post.?op|postoperative|whipple|resection|drain|stent", text):
        return "postoperative_or_device_related"
    if re.search(r"atrophy|calcification|calcified|fatty|fat.invagination|fat replacement", text):
        return "atrophy_calcification_or_benign_fat"
    if str(meta.get("target_presence", "")).lower() == "absent":
        return "normal_or_clean_absent"
    return "other_present"


def outcome_name(gold: bool, pred: bool) -> str:
    if gold and pred:
        return "TP"
    if (not gold) and (not pred):
        return "TN"
    if (not gold) and pred:
        return "FP"
    return "FN"


def predict_detailed(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    kind = cfg["kind"]
    n = arr["gold_present"].shape[0]
    zeros = np.zeros(n, dtype=np.float32)
    false = np.zeros(n, dtype=bool)
    if kind == "weighted_threshold":
        score = lor.weighted_score(arr, cfg)
        pred = score > float(cfg["threshold"])
        return {
            "pred": pred,
            "base_pred": pred.copy(),
            "score": score,
            "threshold": np.full(n, float(cfg["threshold"]), dtype=np.float32),
            "risk": zeros,
            "counter_strength": zeros,
            "positive_strength": zeros,
            "counter": false,
            "dual_counter": false,
            "conflict": false,
            "veto": false,
        }
    if kind == "risk_adjusted_margin":
        score = lor.weighted_score(arr, cfg)
        ctr_strength = lor.max_counter_strength(arr, cfg)
        pos_strength = lor.max_positive_strength(arr, cfg)
        ctr = lor.counter_mask(arr, cfg)
        dual = lor.dual_counter_mask(arr, cfg)
        conflict = (lor.trunk_score(arr, list(cfg.get("routes") or [])) > float(cfg.get("conflict_trunk_min", 0.0))) & ctr
        threshold = (
            float(cfg["threshold"])
            + float(cfg.get("neg_penalty", 0.0)) * ctr_strength
            + float(cfg.get("conflict_penalty", 0.0)) * conflict.astype(np.float32)
            + float(cfg.get("dual_penalty", 0.0)) * dual.astype(np.float32)
            - float(cfg.get("pos_bonus", 0.0)) * pos_strength
        )
        pred = score > threshold
        return {
            "pred": pred,
            "base_pred": pred.copy(),
            "score": score,
            "threshold": threshold.astype(np.float32),
            "risk": threshold.astype(np.float32) - float(cfg["threshold"]),
            "counter_strength": ctr_strength,
            "positive_strength": pos_strength,
            "counter": ctr,
            "dual_counter": dual,
            "conflict": conflict,
            "veto": false,
        }
    if kind == "posthoc_risk_veto":
        base_cfg = dict(cfg["base_cfg"])
        base_detail = predict_detailed(arr, base_cfg)
        score = base_detail["score"]
        ctr_strength = lor.max_counter_strength(arr, base_cfg)
        pos_strength = lor.max_positive_strength(arr, base_cfg)
        ctr = lor.counter_mask(arr, base_cfg)
        dual = lor.dual_counter_mask(arr, base_cfg)
        conflict = (lor.trunk_score(arr, list(base_cfg.get("routes") or [])) > float(base_cfg.get("conflict_trunk_min", 0.0))) & ctr
        risk = (
            float(cfg.get("counter_alpha", 1.0)) * ctr_strength
            + float(cfg.get("conflict_weight", 0.0)) * conflict.astype(np.float32)
            + float(cfg.get("dual_weight", 0.0)) * dual.astype(np.float32)
            - float(cfg.get("pos_credit", 0.0)) * pos_strength
        )
        veto = base_detail["base_pred"] & (risk >= float(cfg["risk_threshold"])) & (score <= float(cfg["score_max"]))
        pred = base_detail["base_pred"] & ~veto
        return {
            "pred": pred,
            "base_pred": base_detail["base_pred"],
            "score": score,
            "threshold": base_detail["threshold"],
            "risk": risk.astype(np.float32),
            "counter_strength": ctr_strength,
            "positive_strength": pos_strength,
            "counter": ctr,
            "dual_counter": dual,
            "conflict": conflict,
            "veto": veto,
        }
    raise ValueError(kind)


def evaluate_cfg(
    label: str,
    cfg: dict[str, Any],
    train_arr: dict[str, np.ndarray],
    final_arr: dict[str, np.ndarray],
    sample_weight: np.ndarray,
    note: str,
) -> dict[str, Any]:
    train_pred, train_extra = lor.predict_config(train_arr, cfg)
    final_pred, final_extra = lor.predict_config(final_arr, cfg)
    train_metric = comp.metric_from_bool(train_arr["gold_present"], train_pred, sample_weight)
    final_metric = comp.metric_from_bool(final_arr["gold_present"], final_pred)
    return {
        "variant": label,
        "config_id": cfg.get("config_id", ""),
        "kind": cfg.get("kind", ""),
        "note": note,
        **metric_prefix("train", train_metric),
        **{f"train_{key}": value for key, value in train_extra.items()},
        **metric_prefix("evaluation", final_metric),
        **{f"evaluation_{key}": value for key, value in final_extra.items()},
    }


def build_plain_rag(
    train_arr: dict[str, np.ndarray],
    final_arr: dict[str, np.ndarray],
    sample_weight: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    configs = plain.candidate_configs()
    train_rows = plain.train_metric_rows(train_arr, configs, sample_weight)
    selected = plain.select_on_train(train_rows)
    final_rows: list[dict[str, Any]] = []
    for row in selected:
        cfg = json.loads(row["config_json"])
        pred, extra = plain.predict_config(final_arr, cfg)
        metric = comp.metric_from_bool(final_arr["gold_present"], pred)
        final_rows.append({**row, **metric_prefix("evaluation", metric), **{f"evaluation_{key}": value for key, value in extra.items()}})

    def pick(**filters: str) -> dict[str, Any] | None:
        rows = final_rows
        for key, value in filters.items():
            rows = [row for row in rows if row.get(key) == value]
        if not rows:
            return None
        return max(rows, key=lambda row: (float(row["train_f1"]), float(row["train_balanced_accuracy"]), float(row["train_specificity"]), -float(row["train_fp"])))

    specs = [
        ("Plain RAG, global train-best", {"selection_scope": "global", "objective": "best_f1"}, "best train F1 across ordinary RAG baselines"),
        ("Trunk-only vanilla RAG", {"baseline": "trunk_only", "objective": "best_f1"}, "three trunk routes, no ROI risk control"),
        ("All-route vanilla RAG", {"baseline": "all_routes", "objective": "best_f1"}, "all routes used as positive evidence"),
        ("Balanced all-route vanilla RAG", {"baseline": "all_routes_roi_balanced", "objective": "best_f1"}, "larger ROI positive weights"),
        ("Merlin-as-positive RAG", {"baseline": "trunk_merlin_positive_equal", "objective": "best_f1"}, "Merlin ROI as ordinary positive route"),
        ("Best route-vote RAG", {"family": "vote_plain_rag", "objective": "best_f1"}, "route vote baseline"),
        ("Best single-route baseline", {"family": "single_route", "objective": "best_f1"}, "best individual retrieval route"),
        ("Plain RAG low-FP recall>=0.85", {"selection_scope": "global", "objective": "min_fp_recall_ge_0p850"}, "train-selected low-FP point"),
        ("Plain RAG max recall spec>=0.80", {"selection_scope": "global", "objective": "max_recall_spec_ge_0p800"}, "specificity-constrained train point"),
    ]
    summary: list[dict[str, Any]] = []
    for role, filters, note in specs:
        row = pick(**filters)
        if row is None:
            continue
        summary.append(
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
    return train_rows, final_rows, summary


def gate_ablation_rows(
    locked_cfg: dict[str, Any],
    train_arr: dict[str, np.ndarray],
    final_arr: dict[str, np.ndarray],
    sample_weight: np.ndarray,
) -> list[dict[str, Any]]:
    base = json.loads(json.dumps(locked_cfg["base_cfg"]))
    rows: list[tuple[str, dict[str, Any], str]] = []
    rows.append(("Locked RCMEA-RAG", json.loads(json.dumps(locked_cfg)), "risk-adjusted margin plus post-hoc ROI veto"))
    rows.append(("No post-hoc veto", base, "base risk-adjusted margin only"))

    weighted = json.loads(json.dumps(base))
    weighted["kind"] = "weighted_threshold"
    weighted["config_id"] = "policy_fixed_weighted_threshold"
    weighted.pop("counter_routes", None)
    weighted.pop("rescue_routes", None)
    rows.append(("Weighted trunk threshold only", weighted, "same trunk score and threshold, no ROI risk terms"))

    no_conflict = json.loads(json.dumps(locked_cfg))
    no_conflict["config_id"] = "policy_no_conflict_penalty"
    no_conflict["base_cfg"]["conflict_penalty"] = 0.0
    rows.append(("No conflict penalty", no_conflict, "remove conflict penalty inside base threshold"))

    no_dual = json.loads(json.dumps(locked_cfg))
    no_dual["config_id"] = "policy_no_dual_penalty"
    no_dual["base_cfg"]["dual_penalty"] = 0.0
    rows.append(("No dual-counter penalty", no_dual, "remove dual-counter penalty inside base threshold"))

    no_cap = json.loads(json.dumps(locked_cfg))
    no_cap["config_id"] = "policy_no_score_cap"
    no_cap["score_max"] = 1_000_000.0
    rows.append(("No post-hoc score cap", no_cap, "allow risk veto regardless of trunk score"))

    tighter = json.loads(json.dumps(locked_cfg))
    tighter["config_id"] = "policy_risk_threshold_2"
    tighter["risk_threshold"] = 2.0
    rows.append(("Tighter post-hoc risk threshold", tighter, "veto only when risk >= 2"))

    no_merlin = json.loads(json.dumps(locked_cfg))
    no_merlin["config_id"] = "policy_no_reference_roi_counter"
    no_merlin["base_cfg"]["counter_routes"] = []
    no_merlin["base_cfg"]["rescue_routes"] = []
    rows.append(("No Merlin ROI counter branch", no_merlin, "remove ROI counter evidence from locked policy"))

    return [evaluate_cfg(label, cfg, train_arr, final_arr, sample_weight, note) for label, cfg, note in rows]


def build_per_case_rows(
    final_arr: dict[str, np.ndarray],
    locked_cfg: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    canonical: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    detail = predict_detailed(final_arr, locked_cfg)
    rows: list[dict[str, Any]] = []
    for i, cid in enumerate(final_arr["case_ids"]):
        cid = str(cid)
        gold = bool(final_arr["gold_present"][i])
        pred = bool(detail["pred"][i])
        meta = metadata.get(cid, {"case_id": cid})
        can = canonical.get(cid, {})
        row: dict[str, Any] = {
            "case_id": cid,
            "gold": "present" if gold else "absent",
            "pred": "present" if pred else "absent",
            "outcome": outcome_name(gold, pred),
            "error_category": category_for_case(meta, can),
            "score": float(detail["score"][i]),
            "threshold": float(detail["threshold"][i]),
            "risk": float(detail["risk"][i]),
            "counter_strength": float(detail["counter_strength"][i]),
            "positive_strength": float(detail["positive_strength"][i]),
            "base_pred": "present" if bool(detail["base_pred"][i]) else "absent",
            "posthoc_veto": bool(detail["veto"][i]),
            "counter_evidence": bool(detail["counter"][i]),
            "dual_counter_evidence": bool(detail["dual_counter"][i]),
            "conflict": bool(detail["conflict"][i]),
            "target_major": meta.get("target_major", ""),
            "target_detail": meta.get("target_detail", ""),
            "rules_locked_category": can.get("rules_locked_category", ""),
        }
        for r_idx, route in enumerate(comp.ROUTES):
            margin = float(final_arr["margins"][i, r_idx])
            if int(final_arr["known"][i, r_idx]) <= 0:
                vote = "unknown"
            elif margin > 0:
                vote = "present"
            elif margin < 0:
                vote = "absent"
            else:
                vote = "tie"
            row[f"{route}_margin"] = margin
            row[f"{route}_present_n"] = int(final_arr["present_n"][i, r_idx])
            row[f"{route}_absent_n"] = int(final_arr["absent_n"][i, r_idx])
            row[f"{route}_vote"] = vote
            row[f"{route}_top1_present"] = bool(final_arr["top1_present"][i, r_idx])
            row[f"{route}_top1_absent"] = bool(final_arr["top1_absent"][i, r_idx])
        rows.append(row)
    return rows


def error_taxonomy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = Counter(row["outcome"] for row in rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["outcome"] in {"FP", "FN"}:
            grouped[(row["outcome"], row["error_category"])].append(row)
    out: list[dict[str, Any]] = []
    for (outcome, category), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], -len(kv[1]), kv[0][1])):
        out.append(
            {
                "outcome": outcome,
                "error_category": category,
                "n": len(items),
                "share_of_outcome": len(items) / totals[outcome] if totals[outcome] else 0.0,
                "example_case_ids": "|".join(item["case_id"] for item in items[:8]),
            }
        )
    return out


def route_support_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for route in comp.ROUTES:
        for outcome in ["TP", "TN", "FP", "FN"]:
            items = [row for row in rows if row["outcome"] == outcome]
            if not items:
                continue
            votes = Counter(row[f"{route}_vote"] for row in items)
            out.append(
                {
                    "route": route,
                    "outcome": outcome,
                    "n": len(items),
                    "present_vote_n": votes.get("present", 0),
                    "absent_vote_n": votes.get("absent", 0),
                    "tie_or_unknown_n": votes.get("tie", 0) + votes.get("unknown", 0),
                    "top1_present_n": sum(bool(row[f"{route}_top1_present"]) for row in items),
                    "top1_absent_n": sum(bool(row[f"{route}_top1_absent"]) for row in items),
                    "mean_margin": float(np.mean([float(row[f"{route}_margin"]) for row in items])),
                    "mean_present_n": float(np.mean([int(row[f"{route}_present_n"]) for row in items])),
                    "mean_absent_n": float(np.mean([int(row[f"{route}_absent_n"]) for row in items])),
                }
            )
    return out


def trunk_pattern_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pattern = "|".join(f"{route}:{row[f'{route}_vote'][0].upper()}" for route in ["fullct", "rsuper", "radgpt", "merlinroi_bal"])
        grouped[pattern].append(row)
    out: list[dict[str, Any]] = []
    for pattern, items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        c = Counter(row["outcome"] for row in items)
        out.append(
            {
                "pattern": pattern,
                "n": len(items),
                "tp": c.get("TP", 0),
                "tn": c.get("TN", 0),
                "fp": c.get("FP", 0),
                "fn": c.get("FN", 0),
                "error_rate": (c.get("FP", 0) + c.get("FN", 0)) / len(items),
            }
        )
    return out


def complementarity_rows(rows: list[dict[str, Any]], locked_metric: dict[str, Any], base_metric: dict[str, Any]) -> list[dict[str, Any]]:
    def subset_stats(label: str, items: list[dict[str, Any]], note: str) -> dict[str, Any]:
        c = Counter(row["outcome"] for row in items)
        return {
            "analysis": label,
            "n": len(items),
            "tp": c.get("TP", 0),
            "tn": c.get("TN", 0),
            "fp": c.get("FP", 0),
            "fn": c.get("FN", 0),
            "error_rate": (c.get("FP", 0) + c.get("FN", 0)) / len(items) if items else 0.0,
            "note": note,
        }

    out: list[dict[str, Any]] = []
    out.append(
        {
            "analysis": "Locked final prediction",
            "n": int(locked_metric["evaluation_n"]),
            "tp": int(locked_metric["evaluation_tp"]),
            "tn": int(locked_metric["evaluation_tn"]),
            "fp": int(locked_metric["evaluation_fp"]),
            "fn": int(locked_metric["evaluation_fn"]),
            "error_rate": 1.0 - float(locked_metric["evaluation_accuracy"]),
            "note": "final locked selected policy output",
        }
    )
    out.append(
        {
            "analysis": "Pre-veto base prediction",
            "n": int(base_metric["evaluation_n"]),
            "tp": int(base_metric["evaluation_tp"]),
            "tn": int(base_metric["evaluation_tn"]),
            "fp": int(base_metric["evaluation_fp"]),
            "fn": int(base_metric["evaluation_fn"]),
            "error_rate": 1.0 - float(base_metric["evaluation_accuracy"]),
            "note": "risk-adjusted base before post-hoc veto",
        }
    )
    vetoed = [row for row in rows if row["posthoc_veto"]]
    out.append(subset_stats("Post-hoc vetoed cases", vetoed, "TN means fixed base FP; FN means over-vetoed positive"))
    counter = [row for row in rows if row["counter_evidence"]]
    out.append(subset_stats("Merlin ROI counter-evidence flagged", counter, "cases where ROI route supplies absent counter-evidence"))
    conflict = [row for row in rows if row["conflict"]]
    out.append(subset_stats("Trunk-positive/ROI-counter conflict", conflict, "conflict cases targeted by risk control"))
    roi_pos = [row for row in rows if float(row["positive_strength"]) > 0]
    out.append(subset_stats("Merlin ROI positive support", roi_pos, "cases where ROI route supports present"))
    trunk_votes = ["fullct_vote", "rsuper_vote", "radgpt_vote"]
    mixed = [row for row in rows if len({row[key] for key in trunk_votes}) > 1]
    all_present = [row for row in rows if all(row[key] == "present" for key in trunk_votes)]
    all_absent = [row for row in rows if all(row[key] == "absent" for key in trunk_votes)]
    out.append(subset_stats("Mixed trunk-route votes", mixed, "retrieval disagreement among full CT, clinical neighbor, and schema routes"))
    out.append(subset_stats("All trunk routes vote present", all_present, "high-support positive consensus"))
    out.append(subset_stats("All trunk routes vote absent", all_absent, "high-support absent consensus"))
    return out


def audit_tables(
    train_arr: dict[str, np.ndarray],
    final_gold: dict[str, str],
    final_coverage: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = read_json(EVALUATION_METADATA_JSON)
    auxiliary = read_json(AUXILIARY_INPUT_DIR / "auxiliary_metadata.json")
    rules = read_json(CANONICAL_INPUT_DIR / "rules_audit.json")
    base_audit = read_json(CANONICAL_INPUT_DIR / "data_audit.json")
    postfix_train = read_json(AUXILIARY_INPUT_DIR / "training_audit.json")
    final_counts = Counter(final_gold.values())
    train_counts = Counter("present" if value else "absent" for value in train_arr["gold_present"])

    protocol = [
        {
            "check": "final evaluation set",
            "value": f"{len(final_gold)} cases; present={final_counts.get('present', 0)}, absent={final_counts.get('absent', 0)}",
            "evidence": "evaluation full_text/test.jsonl",
            "status": "fixed evaluation",
        },
        {
            "check": "train selection pool",
            "value": f"{len(train_arr['gold_present'])} cases; present={train_counts.get('present', 0)}, absent={train_counts.get('absent', 0)}",
            "evidence": "train-only RCMEA-RAG arrays",
            "status": "selection only",
        },
        {
            "check": "operating-point selection",
            "value": "evaluation_used_for_fixed_evaluation_only=true",
            "evidence": "leave-one-route summary and this script",
            "status": "no evaluation threshold selection",
        },
        {
            "check": "query leakage guard",
            "value": f"exclude_all_query_cases={evaluation['splits']['test']['exclude_all_query_cases']}",
            "evidence": "evaluation BUILD_SUMMARY test split",
            "status": "enabled",
        },
        {
            "check": "retrieval coverage",
            "value": ", ".join(f"{route}={final_coverage['route_cases_with_hits'][route]}" for route in comp.ROUTES),
            "evidence": "evaluation retrieval arrays",
            "status": f"allowed_keys={final_coverage['allowed_key_cases']}; extra_keys={final_coverage['extra_key_cases']}",
        },
        {
            "check": "target policy",
            "value": evaluation.get("target_policy", ""),
            "evidence": "evaluation BUILD_SUMMARY",
            "status": "target not rewritten by gate",
        },
        {
            "check": "prompt policy",
            "value": evaluation.get("prompt_policy", ""),
            "evidence": "evaluation BUILD_SUMMARY",
            "status": "prompt/rag evidence rewritten",
        },
        {
            "check": "R-Super source consistency",
            "value": f"test dropped={auxiliary['outputs']['test.jsonl']['dropped_rows']}; train dropped={auxiliary['outputs']['train.jsonl']['dropped_rows']}",
            "evidence": "auxiliary BUILD_SUMMARY",
            "status": "conflicting source-polarity rows removed",
        },
    ]

    label = [
        {
            "stage": "Reviewed evaluation source",
            "n": base_audit["final_n"],
            "present": base_audit["label_counts"]["present"],
            "absent": base_audit["label_counts"]["absent"],
            "removed_or_changed": f"manual_overrides={base_audit['manual_overrides_total']}",
            "audit_note": "canonical reviewed final gold source",
        },
        {
            "stage": "Rules-locked canonical labels",
            "n": rules["rules_locked_included_n"],
            "present": rules["rules_locked_label_counts"]["present"],
            "absent": rules["rules_locked_label_counts"]["absent"],
            "removed_or_changed": f"label_changes={len(rules['binary_label_changes'])}; new_exclusions={rules['rules_locked_new_excluded_n']}",
            "audit_note": "binary rules locked before evaluation",
        },
        {
            "stage": "R-Super-consistent final test",
            "n": auxiliary["outputs"]["test.jsonl"]["rows_out"],
            "present": auxiliary["outputs"]["test.jsonl"]["stats"]["target_presence_present"],
            "absent": auxiliary["outputs"]["test.jsonl"]["stats"]["target_presence_absent"],
            "removed_or_changed": f"dropped_source_conflicts={auxiliary['outputs']['test.jsonl']['dropped_rows']}",
            "audit_note": "evaluation test口径",
        },
        {
            "stage": "Postfix train audit",
            "n": postfix_train["rows"],
            "present": postfix_train["target_presence_counts"]["present"],
            "absent": postfix_train["target_presence_counts"]["absent"],
            "removed_or_changed": f"rules_review_count={postfix_train['rules_review_count']}; target_mismatch={postfix_train['target_mismatch_count']}",
            "audit_note": "no remaining rules-review queue after auxiliary rebuild",
        },
        {
            "stage": "Final evaluation prompt/gate test",
            "n": evaluation["splits"]["test"]["rows"],
            "present": evaluation["splits"]["test"]["target_presence_counts"]["present"],
            "absent": evaluation["splits"]["test"]["target_presence_counts"]["absent"],
            "removed_or_changed": f"gate_fp={int(evaluation['splits']['test']['gate_metric_against_split_target']['fp'])}; gate_fn={int(evaluation['splits']['test']['gate_metric_against_split_target']['fn'])}",
            "audit_note": "targets copied; prompt gate rewritten with selected policy",
        },
    ]
    return protocol, label


def metric_component_ablation() -> list[dict[str, Any]]:
    if not METRIC_COMPONENTS_CSV.exists():
        return []
    rows = read_csv(METRIC_COMPONENTS_CSV)
    keep = {
        "full",
        "no_polarity_gate",
        "critical_state_only",
        "no_field_weights",
        "no_abnormal_weights",
        "synonym_normalized_text",
        "no_synonym_normalization",
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("model") != "report_generator" or row.get("variant") not in keep:
            continue
        out.append(
            {
                "model": row["model"],
                "variant": row["variant"],
                "rows": row["rows"],
                "score_mean": row["score_mean"],
                "auc_correct_gt_incorrect": row["auc_correct_gt_incorrect"],
                "presence_f1": row["presence_f1"],
                "delta_score_mean_vs_full": row["delta_score_mean_vs_full"],
                "delta_auc_vs_full": row["delta_auc_vs_full"],
                "delta_presence_f1_vs_full": row["delta_presence_f1_vs_full"],
            }
        )
    order = [
        "full",
        "no_polarity_gate",
        "critical_state_only",
        "no_field_weights",
        "no_abnormal_weights",
        "synonym_normalized_text",
        "no_synonym_normalization",
    ]
    return sorted(out, key=lambda row: order.index(row["variant"]))


def write_simple_latex(
    path: Path,
    caption: str,
    label: str,
    columns: list[tuple[str, str]],
    rows: list[dict[str, Any]],
    align: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = align or ("l" * len(columns))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header, _ in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(key, "")) for _, key in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tables(
    table_dir: Path,
    locked_row: dict[str, Any],
    plain_summary: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    complement_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> None:
    main_rows = [
        {
            "method": "Locked RCMEA-RAG",
            "selection": "train-selected selected policy",
            "counts": count_str(locked_row, "evaluation"),
            "f1": fmt(locked_row["evaluation_f1"]),
            "spec": fmt(locked_row["evaluation_specificity"]),
            "rec": fmt(locked_row["evaluation_recall"]),
        }
    ]
    include_roles = {
        "Trunk-only vanilla RAG",
        "All-route vanilla RAG",
        "Balanced all-route vanilla RAG",
        "Merlin-as-positive RAG",
        "Best route-vote RAG",
        "Best single-route baseline",
        "Plain RAG low-FP recall>=0.85",
    }
    for row in plain_summary:
        if row["role"] not in include_roles:
            continue
        main_rows.append(
            {
                "method": row["role"],
                "selection": row["objective"],
                "counts": row["evaluation_tp_tn_fp_fn"],
                "f1": fmt(row["evaluation_f1"]),
                "spec": fmt(row["evaluation_specificity"]),
                "rec": fmt(row["evaluation_recall"]),
            }
        )
    write_simple_latex(
        table_dir / "tab_rcmea_rag_evaluation_main_comparison.tex",
        "Evaluation main comparison between locked RCMEA-RAG and train-selected plain RAG baselines.",
        "tab:rcmea_rag_evaluation_main_comparison",
        [("Method", "method"), ("Selection", "selection"), ("TP/TN/FP/FN", "counts"), ("F1", "f1"), ("Spec.", "spec"), ("Rec.", "rec")],
        main_rows,
        align="l l c c c c",
    )

    gate_latex_rows = [
        {
            "variant": row["variant"],
            "counts": count_str(row, "evaluation"),
            "f1": fmt(row["evaluation_f1"]),
            "spec": fmt(row["evaluation_specificity"]),
            "rec": fmt(row["evaluation_recall"]),
            "veto": int(round(float(row.get("evaluation_veto_n", 0) or 0))),
            "note": row["note"],
        }
        for row in gate_rows
    ]
    write_simple_latex(
        table_dir / "tab_rcmea_rag_gate_veto_evaluation.tex",
        "Fixed-policy gate and veto perturbations on evaluation. No variant is re-selected on evaluation.",
        "tab:rcmea_rag_gate_veto_evaluation",
        [("Variant", "variant"), ("TP/TN/FP/FN", "counts"), ("F1", "f1"), ("Spec.", "spec"), ("Rec.", "rec"), ("Veto", "veto"), ("Note", "note")],
        gate_latex_rows,
        align="l c c c c c l",
    )

    write_simple_latex(
        table_dir / "tab_rcmea_rag_protocol_audit_evaluation.tex",
        "Protocol audit for the evaluation RCMEA-RAG evaluation.",
        "tab:rcmea_rag_protocol_audit_evaluation",
        [("Check", "check"), ("Value", "value"), ("Evidence", "evidence"), ("Status", "status")],
        protocol_rows,
        align="l p{0.36\\linewidth} p{0.24\\linewidth} l",
    )

    write_simple_latex(
        table_dir / "tab_rcmea_rag_label_audit_evaluation.tex",
        "Data construction and label audit leading to the evaluation test口径.",
        "tab:rcmea_rag_label_audit_evaluation",
        [("Stage", "stage"), ("N", "n"), ("Present", "present"), ("Absent", "absent"), ("Removed/changed", "removed_or_changed"), ("Note", "audit_note")],
        label_rows,
        align="l r r r l l",
    )

    err_latex = [
        {
            "outcome": row["outcome"],
            "category": row["error_category"],
            "n": row["n"],
            "share": fmt(row["share_of_outcome"]),
            "examples": row["example_case_ids"],
        }
        for row in error_rows
    ]
    write_simple_latex(
        table_dir / "tab_rcmea_rag_error_taxonomy_evaluation.tex",
        "FP/FN error taxonomy for the locked RCMEA-RAG prediction on evaluation.",
        "tab:rcmea_rag_error_taxonomy_evaluation",
        [("Outcome", "outcome"), ("Category", "category"), ("N", "n"), ("Share", "share"), ("Examples", "examples")],
        err_latex,
        align="l l r c p{0.35\\linewidth}",
    )

    comp_keep = [
        "Post-hoc vetoed cases",
        "Merlin ROI counter-evidence flagged",
        "Trunk-positive/ROI-counter conflict",
        "Mixed trunk-route votes",
        "All trunk routes vote present",
        "All trunk routes vote absent",
    ]
    comp_latex = [
        {
            "analysis": row["analysis"],
            "n": row["n"],
            "counts": f"{row['tp']}/{row['tn']}/{row['fp']}/{row['fn']}",
            "err": fmt(row["error_rate"]),
            "note": row["note"],
        }
        for row in complement_rows
        if row["analysis"] in comp_keep
    ]
    write_simple_latex(
        table_dir / "tab_rcmea_rag_retrieval_complementarity_evaluation.tex",
        "Retrieval complementarity diagnostics on evaluation.",
        "tab:rcmea_rag_retrieval_complementarity_evaluation",
        [("Subset", "analysis"), ("N", "n"), ("TP/TN/FP/FN", "counts"), ("Err.", "err"), ("Interpretation", "note")],
        comp_latex,
        align="l r c c p{0.34\\linewidth}",
    )

    metric_latex = [
        {
            "variant": row["variant"],
            "score": fmt(row["score_mean"]),
            "auc": fmt(row["auc_correct_gt_incorrect"]),
            "pf1": fmt(row["presence_f1"]),
            "dscore": fmt(row["delta_score_mean_vs_full"]),
            "dauc": fmt(row["delta_auc_vs_full"]),
        }
        for row in metric_rows
    ]
    write_simple_latex(
        table_dir / "tab_clinical_polarity_metric_component_ablation.tex",
        "Clinical polarity metric component ablation on the 1491-case metric validation set.",
        "tab:clinical_polarity_metric_component_ablation",
        [("Variant", "variant"), ("Mean", "score"), ("AUC", "auc"), ("Presence F1", "pf1"), ("Delta mean", "dscore"), ("Delta AUC", "dauc")],
        metric_latex,
        align="l c c c c c",
    )


def write_markdown_report(
    path: Path,
    locked_row: dict[str, Any],
    plain_summary: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    complement_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# RCMEA-RAG Evaluation Evidence Pack",
        "",
        "All operating points are train-selected or fixed perturbations of locked selected policy. evaluation is fixed evaluation only.",
        "",
        "## Main Comparison",
        "",
        "| method | final TP/TN/FP/FN | F1 | Spec | Recall |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| Locked RCMEA-RAG | {count_str(locked_row, 'evaluation')} | {fmt(locked_row['evaluation_f1'], 4)} | {fmt(locked_row['evaluation_specificity'], 4)} | {fmt(locked_row['evaluation_recall'], 4)} |",
    ]
    for row in plain_summary:
        lines.append(
            f"| {row['role']} | {row['evaluation_tp_tn_fp_fn']} | {fmt(row['evaluation_f1'], 4)} | {fmt(row['evaluation_specificity'], 4)} | {fmt(row['evaluation_recall'], 4)} |"
        )
    lines.extend(["", "## Gate/Veto Fixed Perturbations", "", "| variant | final TP/TN/FP/FN | F1 | Spec | Recall | note |", "| --- | --- | ---: | ---: | ---: | --- |"])
    for row in gate_rows:
        lines.append(
            f"| {row['variant']} | {count_str(row, 'evaluation')} | {fmt(row['evaluation_f1'], 4)} | {fmt(row['evaluation_specificity'], 4)} | {fmt(row['evaluation_recall'], 4)} | {row['note']} |"
        )
    lines.extend(["", "## Protocol Audit", "", "| check | value | status |", "| --- | --- | --- |"])
    for row in protocol_rows:
        lines.append(f"| {row['check']} | {row['value']} | {row['status']} |")
    lines.extend(["", "## Label Audit", "", "| stage | n | present | absent | removed/changed |", "| --- | ---: | ---: | ---: | --- |"])
    for row in label_rows:
        lines.append(f"| {row['stage']} | {row['n']} | {row['present']} | {row['absent']} | {row['removed_or_changed']} |")
    lines.extend(["", "## FP/FN Error Taxonomy", "", "| outcome | category | n | share | examples |", "| --- | --- | ---: | ---: | --- |"])
    for row in error_rows:
        lines.append(f"| {row['outcome']} | {row['error_category']} | {row['n']} | {fmt(row['share_of_outcome'], 3)} | {row['example_case_ids']} |")
    lines.extend(["", "## Retrieval Complementarity", "", "| analysis | n | TP/TN/FP/FN | error rate | note |", "| --- | ---: | --- | ---: | --- |"])
    for row in complement_rows:
        lines.append(f"| {row['analysis']} | {row['n']} | {row['tp']}/{row['tn']}/{row['fp']}/{row['fn']} | {fmt(row['error_rate'], 3)} | {row['note']} |")
    lines.extend(["", "## Metric Component Ablation", "", "| variant | mean | AUC | presence F1 | delta mean |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in metric_rows:
        lines.append(f"| {row['variant']} | {fmt(row['score_mean'], 3)} | {fmt(row['auc_correct_gt_incorrect'], 3)} | {fmt(row['presence_f1'], 3)} | {fmt(row['delta_score_mean_vs_full'], 3)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    final_gold = lor.load_gold_from_jsonl(args.evaluation_records_jsonl)
    locked_cfg = lor.load_locked_config(args.evaluation_metadata_json)
    metadata = load_case_metadata(args.evaluation_records_jsonl)
    canonical = load_canonical_map(CANONICAL_INPUT_DIR / "canonical_labels.csv")

    inputs = focused.build_inputs()
    train_arr, train_coverage = comp.build_arrays_for_fold(
        lor.SCENARIO,
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
        lor.SCENARIO,
        final_gold,
        inputs,
        args.embedding_batch_size,
        args.candidate_pool,
    )
    sample_weight = comp.sample_weights_for_scenario(train_arr, lor.SCENARIO)

    locked_row = evaluate_cfg("Locked RCMEA-RAG", locked_cfg, train_arr, final_arr, sample_weight, "locked selected policy")
    plain_train_rows, plain_final_rows, plain_summary = build_plain_rag(train_arr, final_arr, sample_weight)
    gate_rows = gate_ablation_rows(locked_cfg, train_arr, final_arr, sample_weight)
    per_case = build_per_case_rows(final_arr, locked_cfg, metadata, canonical)
    err_rows = error_taxonomy(per_case)
    support_rows = route_support_summary(per_case)
    pattern_rows = trunk_pattern_summary(per_case)
    base_row = next(row for row in gate_rows if row["variant"] == "No post-hoc veto")
    complement_rows = complementarity_rows(per_case, locked_row, base_row)
    protocol_rows, label_rows = audit_tables(train_arr, final_gold, final_coverage)
    metric_rows = metric_component_ablation()

    write_csv(args.output_dir / "plain_rag_train_grid_evaluation_protocol.csv", plain_train_rows)
    write_csv(args.output_dir / "plain_rag_train_selected_evaluation_fixed_eval.csv", plain_final_rows)
    write_csv(args.output_dir / "plain_rag_baseline_summary_evaluation.csv", plain_summary)
    write_csv(args.output_dir / "gate_veto_ablation_evaluation.csv", gate_rows)
    write_csv(args.output_dir / "locked_policy_per_case_predictions_evaluation.csv", per_case)
    write_csv(args.output_dir / "fp_fn_error_taxonomy_evaluation.csv", err_rows)
    write_csv(args.output_dir / "route_support_by_outcome_evaluation.csv", support_rows)
    write_csv(args.output_dir / "route_vote_pattern_summary_evaluation.csv", pattern_rows)
    write_csv(args.output_dir / "retrieval_complementarity_summary_evaluation.csv", complement_rows)
    write_csv(args.output_dir / "protocol_audit_evaluation.csv", protocol_rows)
    write_csv(args.output_dir / "data_label_audit_evaluation.csv", label_rows)
    write_csv(args.output_dir / "metric_component_ablation_summary.csv", metric_rows)

    write_tables(
        args.table_dir,
        locked_row,
        plain_summary,
        gate_rows,
        protocol_rows,
        label_rows,
        err_rows,
        complement_rows,
        metric_rows,
    )
    write_markdown_report(
        args.output_dir / "RESULTS.md",
        locked_row,
        plain_summary,
        gate_rows,
        protocol_rows,
        label_rows,
        err_rows,
        complement_rows,
        metric_rows,
    )

    summary = {
        "evaluation_cases": len(final_gold),
        "evaluation_presence": dict(Counter(final_gold.values())),
        "train_cases": int(train_arr["gold_present"].shape[0]),
        "train_coverage": train_coverage,
        "final_coverage": final_coverage,
        "locked": locked_row,
        "outputs": {
            "results_md": str(args.output_dir / "RESULTS.md"),
            "plain_summary": str(args.output_dir / "plain_rag_baseline_summary_evaluation.csv"),
            "gate_veto": str(args.output_dir / "gate_veto_ablation_evaluation.csv"),
            "per_case": str(args.output_dir / "locked_policy_per_case_predictions_evaluation.csv"),
            "error_taxonomy": str(args.output_dir / "fp_fn_error_taxonomy_evaluation.csv"),
            "route_support": str(args.output_dir / "route_support_by_outcome_evaluation.csv"),
            "route_patterns": str(args.output_dir / "route_vote_pattern_summary_evaluation.csv"),
            "protocol_audit": str(args.output_dir / "protocol_audit_evaluation.csv"),
            "label_audit": str(args.output_dir / "data_label_audit_evaluation.csv"),
            "metric_ablation": str(args.output_dir / "metric_component_ablation_summary.csv"),
            "tables_dir": str(args.table_dir),
        },
        "no_fabrication": "all numeric values are computed from local artifacts or copied from local audit summaries",
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"results_md": summary["outputs"]["results_md"], "tables_dir": summary["outputs"]["tables_dir"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
