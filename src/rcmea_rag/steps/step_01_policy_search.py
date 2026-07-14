#!/usr/bin/env python3
"""Comprehensive train-only RAG gate policy search.

This script is intentionally stricter than the earlier evaluation-oriented
evaluation scripts:
- evaluation labels are never read;
- evaluation val/test rows are not used for threshold/weight/config selection;
- route hit banks are restricted to training-derived case IDs;
- OOF validation removes the held-out fold from the retrieval bank.

It tests rule-based, adaptive sparse/NSA-style, veto/rescue, and lightweight
learned-router variants under several train-only distribution assumptions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


INPUT_ROOT = Path(os.environ.get("RCMEA_INPUT_ROOT", "inputs")).resolve()
OUTPUT_ROOT = Path(os.environ.get("RCMEA_OUTPUT_ROOT", "outputs")).resolve()
OUT_DIR = OUTPUT_ROOT / "step_01_policy_search"

TRAIN_RECORDS_JSONL = INPUT_ROOT / "training_records.jsonl"
AUXILIARY_TRAIN_JSONL = INPUT_ROOT / "auxiliary_training_records.jsonl"
FULLCT_ROUTE = INPUT_ROOT / "fullct_route_hits.jsonl"
RSUPER_ROUTE = INPUT_ROOT / "structured_route_hits.jsonl"
RADGPT_ROUTE = INPUT_ROOT / "regional_route_hits.jsonl"
MERLIN_BAL_KEY_NPZ = INPUT_ROOT / "reference_roi_keys.npz"

ROUTES = ["fullct", "rsuper", "radgpt", "merlinroi_bal"]
TRUNK_ROUTES = ["fullct", "rsuper", "radgpt"]
ROI_ROUTES = ["merlinroi_bal"]
ROUTE_INDEX = {route: idx for idx, route in enumerate(ROUTES)}
RECALL_FLOORS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25]
THRESHOLDS = [round(-15.0 + 0.5 * idx, 3) for idx in range(65)]
COARSE_THRESHOLDS = [round(-14.0 + idx, 3) for idx in range(29)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--candidate-pool", type=int, default=120)
    parser.add_argument("--max-learned-iter", type=int, default=400)
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use fewer distribution scenarios for smoke testing.")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_case_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("case:"):
        text = text.split(":", 1)[1]
    match = re.search(r"AC[0-9a-fA-F]+", text)
    return match.group(0) if match else text


def fold_for_case(case_id: str, folds: int) -> int:
    digest = hashlib.md5(case_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


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


def config_json(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_original_gold(path: Path) -> dict[str, str]:
    gold: dict[str, str] = {}
    for row in iter_jsonl(path):
        cid = clean_case_id(row.get("case_id"))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        presence = str(metadata.get("lesion_presence") or "").strip().lower()
        if cid and presence in {"present", "absent"}:
            gold[cid] = presence
    return gold


def load_audit_gold(path: Path) -> dict[str, str]:
    gold: dict[str, str] = {}
    for row in iter_jsonl(path):
        cid = clean_case_id(row.get("case_id"))
        presence = str(row.get("schema_presence") or row.get("target_presence") or "").strip().lower()
        if cid and presence in {"present", "absent"}:
            gold[cid] = presence
    return gold


def hit_score(hit: dict[str, Any]) -> float:
    for key in ("schema_score", "field_score", "retrieval_score", "score", "clinical_similarity"):
        value = hit.get(key)
        if value in (None, ""):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return max(score, 0.0)
    return 1.0


def load_raw_standard_route(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(path):
        qid = clean_case_id(row.get("case_id"))
        hits = row.get("nonself_top_hits") or row.get("top_non_self_hits") or row.get("top_hits") or []
        rows: list[dict[str, Any]] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            hid = clean_case_id(hit.get("case_id"))
            if not hid or hid == qid:
                continue
            rows.append({"case_id": hid, "rank": int(hit.get("rank") or len(rows) + 1), "score": hit_score(hit)})
        out[qid] = rows
    return out


def standard_stats(
    qid: str,
    raw_hits: dict[str, list[dict[str, Any]]],
    labels: dict[str, str],
    allowed_cases: set[str],
    exclude_cases: set[str],
    route: str,
) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in raw_hits.get(qid, []):
        hid = clean_case_id(hit.get("case_id"))
        if not hid or hid == qid or hid in seen or hid in exclude_cases or hid not in allowed_cases:
            continue
        label = labels.get(hid)
        if label not in {"present", "absent"}:
            continue
        seen.add(hid)
        kept.append({"route": route, "case_id": hid, "presence": label, "rank": len(kept) + 1})
        if len(kept) >= 5:
            break
    return hits_to_stats(kept)


def hits_to_stats(hits: list[dict[str, Any]]) -> dict[str, Any]:
    present_n = sum(1 for hit in hits if hit.get("presence") == "present")
    absent_n = sum(1 for hit in hits if hit.get("presence") == "absent")
    return {
        "known": len(hits),
        "present_n": present_n,
        "absent_n": absent_n,
        "margin_n": present_n - absent_n,
        "top1": hits[0]["presence"] if hits else "",
    }


def load_npz(path: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    embeddings = payload["embeddings"].astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    rows = json.loads(str(payload["rows_json"]))
    metadata = json.loads(str(payload["metadata_json"])) if "metadata_json" in payload else {}
    if len(rows) != embeddings.shape[0]:
        raise RuntimeError(f"{path}: rows_json length != embeddings rows")
    return embeddings, rows, metadata


def choose_query_indices(rows: list[dict[str, Any]], case_ids: set[str]) -> dict[str, int]:
    by_case: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        cid = clean_case_id(row.get("case_id"))
        if cid in case_ids:
            by_case.setdefault(cid, []).append(idx)
    chosen: dict[str, int] = {}
    for cid, indices in by_case.items():
        indices.sort(
            key=lambda i: (
                "__" in str(rows[i].get("example_id") or ""),
                bool(rows[i].get("is_augmented")),
                str(rows[i].get("augmentation_mode") or ""),
                int(rows[i].get("row_index") or i),
            )
        )
        chosen[cid] = indices[0]
    return chosen


def is_augmented_row(row: dict[str, Any]) -> bool:
    example_id = str(row.get("example_id") or "")
    return "__" in example_id or bool(row.get("is_augmented")) or bool(row.get("augmentation_mode"))


def merlin_row_label(row: dict[str, Any], original_gold: dict[str, str], extra_gold: dict[str, str]) -> str:
    cid = clean_case_id(row.get("case_id"))
    if cid in original_gold:
        return original_gold[cid]
    if cid in extra_gold:
        return extra_gold[cid]
    presence = str(row.get("lesion_presence") or "").strip().lower()
    return presence if presence in {"present", "absent"} else ""


def build_embedding_stats_for_queries(
    route: str,
    query_case_ids: list[str],
    query_index_by_case: dict[str, int],
    query_embeddings_source: np.ndarray,
    key_embeddings: np.ndarray,
    key_rows: list[dict[str, Any]],
    key_labels: list[str],
    allowed_cases: set[str],
    exclude_cases: set[str],
    allow_augmented_keys: bool,
    batch_size: int,
    candidate_pool: int,
) -> dict[str, dict[str, Any]]:
    key_case_ids = np.asarray([clean_case_id(row.get("case_id")) for row in key_rows], dtype=object)
    valid = np.zeros(len(key_rows), dtype=bool)
    for idx, (row, label) in enumerate(zip(key_rows, key_labels)):
        cid = str(key_case_ids[idx])
        if cid not in allowed_cases or cid in exclude_cases or label not in {"present", "absent"}:
            continue
        if not allow_augmented_keys and is_augmented_row(row):
            continue
        valid[idx] = True

    stats_by_case = {cid: {"known": 0, "present_n": 0, "absent_n": 0, "margin_n": 0, "top1": ""} for cid in query_case_ids}
    query_items = [(cid, query_index_by_case[cid]) for cid in query_case_ids if cid in query_index_by_case]
    if not query_items or not np.any(valid):
        return stats_by_case

    key_t = key_embeddings.T
    for start in range(0, len(query_items), batch_size):
        chunk = query_items[start : start + batch_size]
        qidx = [idx for _, idx in chunk]
        scores = query_embeddings_source[qidx] @ key_t
        for local, (qid, _) in enumerate(chunk):
            allowed = valid.copy()
            allowed &= key_case_ids != qid
            row_scores = scores[local].copy()
            row_scores[~allowed] = -np.inf
            finite_n = int(np.isfinite(row_scores).sum())
            if finite_n <= 0:
                continue
            pool = min(max(candidate_pool, 5), finite_n)
            if pool >= len(row_scores):
                candidate_idx = np.argsort(-row_scores)
            else:
                candidate_idx = np.argpartition(-row_scores, pool - 1)[:pool]
                candidate_idx = candidate_idx[np.argsort(-row_scores[candidate_idx])]
            hits: list[dict[str, Any]] = []
            seen_cases: set[str] = set()
            for key_idx in candidate_idx:
                hid = str(key_case_ids[int(key_idx)])
                if not hid or hid in seen_cases:
                    continue
                score = float(row_scores[int(key_idx)])
                if not math.isfinite(score):
                    continue
                seen_cases.add(hid)
                hits.append(
                    {
                        "route": route,
                        "case_id": hid,
                        "presence": key_labels[int(key_idx)],
                        "rank": len(hits) + 1,
                        "score": score,
                    }
                )
                if len(hits) >= 5:
                    break
            stats_by_case[qid] = hits_to_stats(hits)
    return stats_by_case


def metric_from_bool(y: np.ndarray, pred: np.ndarray, sample_weight: np.ndarray | None = None) -> dict[str, float]:
    y = y.astype(bool)
    pred = pred.astype(bool)
    if sample_weight is None:
        w = np.ones(y.shape[0], dtype=np.float64)
    else:
        w = sample_weight.astype(np.float64)
    tp = float(np.sum(w[y & pred]))
    tn = float(np.sum(w[(~y) & (~pred)]))
    fp = float(np.sum(w[(~y) & pred]))
    fn = float(np.sum(w[y & (~pred)]))
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def metric_prefix(prefix: str, metric: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metric.items()}


def examples_to_arrays(case_ids: list[str], gold: dict[str, str], fold_ids: np.ndarray, stats_by_route: dict[str, dict[str, dict[str, Any]]]) -> dict[str, np.ndarray]:
    n = len(case_ids)
    margins = np.zeros((n, len(ROUTES)), dtype=np.float32)
    known = np.zeros((n, len(ROUTES)), dtype=np.int16)
    present_n = np.zeros((n, len(ROUTES)), dtype=np.int16)
    absent_n = np.zeros((n, len(ROUTES)), dtype=np.int16)
    top1_absent = np.zeros((n, len(ROUTES)), dtype=bool)
    top1_present = np.zeros((n, len(ROUTES)), dtype=bool)
    y = np.asarray([gold[cid] == "present" for cid in case_ids], dtype=bool)
    for i, cid in enumerate(case_ids):
        for route, ridx in ROUTE_INDEX.items():
            stats = stats_by_route[route].get(cid) or {"known": 0, "present_n": 0, "absent_n": 0, "margin_n": 0, "top1": ""}
            margins[i, ridx] = float(stats["margin_n"])
            known[i, ridx] = int(stats["known"])
            present_n[i, ridx] = int(stats["present_n"])
            absent_n[i, ridx] = int(stats["absent_n"])
            top1_absent[i, ridx] = stats["top1"] == "absent"
            top1_present[i, ridx] = stats["top1"] == "present"
    return {
        "case_ids": np.asarray(case_ids, dtype=object),
        "fold_ids": fold_ids,
        "gold_present": y,
        "margins": margins,
        "known": known,
        "present_n": present_n,
        "absent_n": absent_n,
        "top1_absent": top1_absent,
        "top1_present": top1_present,
    }


def route_indices(routes: list[str]) -> list[int]:
    return [ROUTE_INDEX[route] for route in routes]


def weights_vector(weights: dict[str, float]) -> np.ndarray:
    return np.asarray([float(weights.get(route, 0.0)) for route in ROUTES], dtype=np.float32)


def weighted_score(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    idx = route_indices(cfg["routes"])
    w = weights_vector(cfg.get("weights") or {})
    return arr["margins"][:, idx] @ w[idx]


def sparse_score(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    idx = route_indices(cfg["routes"])
    margins = arr["margins"][:, idx]
    known = arr["known"][:, idx]
    order = np.argsort(-(np.abs(margins) * 10.0 + known * 0.01), axis=1)
    selected = order[:, : int(cfg["topk"])]
    selected_idx = np.take(np.asarray(idx), selected)
    full_w = weights_vector(cfg.get("weights") or {})
    selected_w = full_w[selected_idx]
    selected_m = np.take_along_axis(arr["margins"], selected_idx, axis=1)
    return np.sum(selected_w * selected_m, axis=1)


def nsa_score(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    idx = route_indices(cfg["routes"])
    margins = arr["margins"][:, idx]
    known = arr["known"][:, idx]
    order = np.argsort(-(np.abs(margins) * 10.0 + known * 0.01), axis=1)
    ordered_idx = np.take(np.asarray(idx), order)
    full_w = weights_vector(cfg.get("weights") or {})
    ordered_scores = np.take_along_axis(arr["margins"], ordered_idx, axis=1) * full_w[ordered_idx]
    cumsum = np.cumsum(ordered_scores, axis=1)
    min_k = int(cfg["min_k"])
    stop = np.abs(cumsum) >= float(cfg["stop_margin"])
    if min_k > 1:
        stop[:, : min_k - 1] = False
    any_stop = np.any(stop, axis=1)
    first_stop = np.argmax(stop, axis=1)
    chosen = np.where(any_stop, first_stop, cumsum.shape[1] - 1)
    return cumsum[np.arange(cumsum.shape[0]), chosen]


def route_agreement_score(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    idx = route_indices(cfg["routes"])
    return np.sum(arr["margins"][:, idx] >= int(cfg["positive_cut"]), axis=1).astype(np.float32)


def roi_veto_mask(arr: dict[str, np.ndarray], cfg: dict[str, Any], score: np.ndarray) -> np.ndarray:
    idx = route_indices(cfg.get("veto_routes") or [])
    if not idx:
        return np.zeros(score.shape[0], dtype=bool)
    neg_margin = arr["absent_n"][:, idx] - arr["present_n"][:, idx]
    mask = (arr["known"][:, idx] > 0) & (neg_margin >= int(cfg["veto_neg_margin"]))
    mask &= arr["present_n"][:, idx] <= int(cfg["veto_present_max"])
    if bool(cfg.get("veto_require_top1_absent")):
        mask &= arr["top1_absent"][:, idx]
    return np.any(mask, axis=1) & (score <= float(cfg["veto_score_max"]))


def roi_rescue_mask(arr: dict[str, np.ndarray], cfg: dict[str, Any], score: np.ndarray) -> np.ndarray:
    idx = route_indices(cfg.get("rescue_routes") or [])
    if not idx:
        return np.zeros(score.shape[0], dtype=bool)
    pos_margin = arr["present_n"][:, idx] - arr["absent_n"][:, idx]
    mask = (arr["known"][:, idx] > 0) & (pos_margin >= int(cfg["rescue_pos_margin"]))
    mask &= arr["absent_n"][:, idx] <= int(cfg["rescue_absent_max"])
    return (np.sum(mask, axis=1) >= int(cfg.get("rescue_route_min", 1))) & (score >= float(cfg["rescue_score_min"]))


def predict_config(arr: dict[str, np.ndarray], cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    kind = cfg["kind"]
    if kind in {
        "weighted_threshold",
        "roi_veto",
        "roi_rescue",
        "hybrid_veto_rescue",
        "adaptive_threshold",
        "adaptive_roi_veto",
    }:
        score = weighted_score(arr, cfg)
    elif kind == "sparse_topk":
        score = sparse_score(arr, cfg)
    elif kind == "nsa_dynamic_sparse":
        score = nsa_score(arr, cfg)
    elif kind == "route_agreement":
        score = route_agreement_score(arr, cfg)
    else:
        raise ValueError(kind)

    threshold = np.full(score.shape[0], float(cfg["threshold"]), dtype=np.float32)
    if kind in {"adaptive_threshold", "adaptive_roi_veto"}:
        roi_idx = route_indices(cfg.get("adaptive_routes") or ROI_ROUTES)
        roi_neg = np.any(
            (arr["known"][:, roi_idx] > 0)
            & ((arr["absent_n"][:, roi_idx] - arr["present_n"][:, roi_idx]) >= int(cfg["roi_neg_margin"])),
            axis=1,
        )
        roi_pos = np.any(
            (arr["known"][:, roi_idx] > 0)
            & ((arr["present_n"][:, roi_idx] - arr["absent_n"][:, roi_idx]) >= int(cfg["roi_pos_margin"])),
            axis=1,
        )
        conflict = np.any(arr["margins"][:, route_indices(TRUNK_ROUTES)] > 0, axis=1) & roi_neg
        threshold = threshold + roi_neg.astype(np.float32) * float(cfg["roi_neg_penalty"])
        threshold = threshold - roi_pos.astype(np.float32) * float(cfg["roi_pos_bonus"])
        threshold = threshold + conflict.astype(np.float32) * float(cfg["conflict_penalty"])

    pred = score > threshold
    veto = np.zeros(pred.shape[0], dtype=bool)
    rescue = np.zeros(pred.shape[0], dtype=bool)
    if kind in {"roi_veto", "hybrid_veto_rescue", "adaptive_roi_veto"}:
        veto = pred & roi_veto_mask(arr, cfg, score)
        pred = pred & ~veto
    if kind in {"roi_rescue", "hybrid_veto_rescue"}:
        rescue = (~pred) & (~veto) & roi_rescue_mask(arr, cfg, score)
        pred = pred | rescue
    return pred, {"veto_n": int(np.sum(veto)), "rescue_n": int(np.sum(rescue))}


def all_route_subsets() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {
        "trunk": TRUNK_ROUTES,
        "trunk_merlinroi_bal": TRUNK_ROUTES + ["merlinroi_bal"],
        "all_routes": ROUTES,
    }
    for n in range(1, len(ROUTES) + 1):
        for combo in itertools.combinations(ROUTES, n):
            out.setdefault("subset_" + "_".join(combo), list(combo))
    return out


def weight_presets() -> dict[str, dict[str, float]]:
    return {
        "unit": {route: 1.0 for route in ROUTES},
        "weighted_routes": {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75, "merlinroi_bal": 0.25},
        "rsuper_heavy_roi_light": {"fullct": 1.0, "rsuper": 2.0, "radgpt": 0.75, "merlinroi_bal": 0.25},
        "roi_equal": {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75, "merlinroi_bal": 1.0},
        "roi_veto_light": {"fullct": 1.0, "rsuper": 1.5, "radgpt": 0.75, "merlinroi_bal": 0.0},
    }


def candidate_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    def add(cfg: dict[str, Any]) -> None:
        row = dict(cfg)
        row["config_id"] = f"R{len(configs) + 1:06d}"
        configs.append(row)

    subsets = all_route_subsets()
    weights_all = weight_presets()
    for set_name, routes in subsets.items():
        for weight_name, full_weights in weights_all.items():
            weights = {route: full_weights.get(route, 1.0) for route in routes}
            for threshold in THRESHOLDS:
                add({"kind": "weighted_threshold", "preset": f"{set_name}_{weight_name}", "routes": routes, "weights": weights, "threshold": threshold})

    sparse_sets = {name: subsets[name] for name in ["trunk", "trunk_merlinroi_bal", "all_routes"]}
    for set_name, routes in sparse_sets.items():
        for weight_name in ["unit", "weighted_routes", "rsuper_heavy_roi_light"]:
            full_weights = weights_all[weight_name]
            weights = {route: full_weights.get(route, 1.0) for route in routes}
            for topk in range(1, len(routes) + 1):
                for threshold in COARSE_THRESHOLDS:
                    add({"kind": "sparse_topk", "preset": f"{set_name}_{weight_name}", "routes": routes, "weights": weights, "topk": topk, "threshold": threshold})
            for min_k in [1, 2]:
                for stop_margin in [1.0, 2.0, 3.0, 4.0, 5.0]:
                    for threshold in COARSE_THRESHOLDS:
                        add(
                            {
                                "kind": "nsa_dynamic_sparse",
                                "preset": f"{set_name}_{weight_name}",
                                "routes": routes,
                                "weights": weights,
                                "min_k": min_k,
                                "stop_margin": stop_margin,
                                "threshold": threshold,
                            }
                        )

    for base_name in ["trunk", "all_routes", "trunk_merlinroi_bal"]:
        routes = subsets[base_name]
        for weight_name in ["weighted_routes", "rsuper_heavy_roi_light", "roi_equal"]:
            weights = {route: weights_all[weight_name].get(route, 1.0) for route in routes}
            for threshold in [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]:
                for roi_neg_penalty in [1.0, 2.0, 3.0, 4.0]:
                    for roi_pos_bonus in [0.0, 1.0, 2.0]:
                        for conflict_penalty in [0.0, 1.0, 2.0]:
                            add(
                                {
                                    "kind": "adaptive_threshold",
                                    "preset": f"{base_name}_{weight_name}_roi_adaptive",
                                    "routes": routes,
                                    "weights": weights,
                                    "threshold": threshold,
                                    "roi_neg_margin": 2,
                                    "roi_pos_margin": 2,
                                    "roi_neg_penalty": roi_neg_penalty,
                                    "roi_pos_bonus": roi_pos_bonus,
                                    "conflict_penalty": conflict_penalty,
                                }
                            )

    veto_sets = {"roi_veto": ROI_ROUTES}
    rescue_sets = {"roi_rescue": ROI_ROUTES}
    for base_name in ["trunk", "all_routes"]:
        routes = subsets[base_name]
        for weight_name in ["weighted_routes", "rsuper_heavy_roi_light"]:
            weights = {route: weights_all[weight_name].get(route, 1.0) for route in routes}
            for threshold in [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]:
                for veto_name, veto_routes in veto_sets.items():
                    for neg_margin in [2, 3, 4, 5]:
                        for present_max in [0, 1]:
                            for top1_absent in [False, True]:
                                for veto_score_max in [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0]:
                                    add(
                                        {
                                            "kind": "roi_veto",
                                            "preset": f"{base_name}_{weight_name}_{veto_name}",
                                            "routes": routes,
                                            "weights": weights,
                                            "threshold": threshold,
                                            "veto_routes": veto_routes,
                                            "veto_neg_margin": neg_margin,
                                            "veto_present_max": present_max,
                                            "veto_require_top1_absent": top1_absent,
                                            "veto_score_max": veto_score_max,
                                        }
                                    )
                for rescue_name, rescue_routes in rescue_sets.items():
                    for pos_margin in [2, 3, 4, 5]:
                        for absent_max in [0, 1]:
                            for rescue_score_min in [-8.0, -4.0, -2.0, -1.0, 0.0]:
                                add(
                                    {
                                        "kind": "roi_rescue",
                                        "preset": f"{base_name}_{weight_name}_{rescue_name}",
                                        "routes": routes,
                                        "weights": weights,
                                        "threshold": threshold,
                                        "rescue_routes": rescue_routes,
                                        "rescue_pos_margin": pos_margin,
                                        "rescue_absent_max": absent_max,
                                        "rescue_score_min": rescue_score_min,
                                        "rescue_route_min": 1,
                                    }
                                )
                for neg_margin in [2, 3, 4, 5]:
                    for pos_margin in [3, 4, 5]:
                        add(
                            {
                                "kind": "hybrid_veto_rescue",
                                "preset": f"{base_name}_{weight_name}_roi_veto_rescue",
                                "routes": routes,
                                "weights": weights,
                                "threshold": threshold,
                                "veto_routes": ROI_ROUTES,
                                "veto_neg_margin": neg_margin,
                                "veto_present_max": 0,
                                "veto_require_top1_absent": True,
                                "veto_score_max": 2.0,
                                "rescue_routes": ["merlinroi_bal"],
                                "rescue_pos_margin": pos_margin,
                                "rescue_absent_max": 0,
                                "rescue_score_min": -2.0,
                                "rescue_route_min": 1,
                            }
                        )

    for set_name, routes in sparse_sets.items():
        for positive_cut in [1, 2, 3]:
            for route_min in range(1, len(routes) + 1):
                add({"kind": "route_agreement", "preset": set_name, "routes": routes, "weights": {}, "threshold": route_min - 0.5, "positive_cut": positive_cut})
    return configs


def scenario_rows(quick: bool) -> list[dict[str, Any]]:
    bank_modes = [
        {"bank_mode": "base_training_bank", "include_extra": "none", "allow_augmented_keys": False},
        {"bank_mode": "augmented_training_bank", "include_extra": "none", "allow_augmented_keys": True},
        {"bank_mode": "extended_absent_bank", "include_extra": "absent", "allow_augmented_keys": False},
        {"bank_mode": "augmented_absent_bank", "include_extra": "absent", "allow_augmented_keys": True},
        {"bank_mode": "default_training_bank", "include_extra": "all", "allow_augmented_keys": True},
    ]
    weight_modes = [
        {"weight_mode": "ratio_1to1", "pos_weight": 1.0, "neg_weight": 1.0},
        {"weight_mode": "neg_2x", "pos_weight": 1.0, "neg_weight": 2.0},
        {"weight_mode": "neg_3x", "pos_weight": 1.0, "neg_weight": 3.0},
        {"weight_mode": "neg_4x", "pos_weight": 1.0, "neg_weight": 4.0},
        {"weight_mode": "pos_aug_2x", "pos_weight": 2.0, "neg_weight": 1.0},
        {"weight_mode": "pos_aug_3x", "pos_weight": 3.0, "neg_weight": 1.0},
    ]
    if quick:
        bank_modes = [bank_modes[0], bank_modes[3]]
        weight_modes = [weight_modes[0], weight_modes[2], weight_modes[4]]
    rows: list[dict[str, Any]] = []
    for bank in bank_modes:
        for weight in weight_modes:
            rows.append({**bank, **weight, "scenario": f"{bank['bank_mode']}__{weight['weight_mode']}"})
    return rows


def bank_key(scenario: dict[str, Any]) -> tuple[str, str, bool]:
    return (str(scenario["bank_mode"]), str(scenario["include_extra"]), bool(scenario["allow_augmented_keys"]))


def build_allowed_labels(original_gold: dict[str, str], audit_gold: dict[str, str], include_extra: str) -> dict[str, str]:
    labels = dict(original_gold)
    extras = {cid: label for cid, label in audit_gold.items() if cid not in original_gold}
    if include_extra == "absent":
        labels.update({cid: label for cid, label in extras.items() if label == "absent"})
    elif include_extra == "all":
        labels.update(extras)
    elif include_extra != "none":
        raise ValueError(include_extra)
    return labels


def build_arrays_for_fold(
    scenario: dict[str, Any],
    heldout_fold: int | None,
    case_ids: list[str],
    gold: dict[str, str],
    audit_gold: dict[str, str],
    fold_ids: np.ndarray,
    raw_routes: dict[str, dict[str, list[dict[str, Any]]]],
    mer: dict[str, Any],
    batch_size: int,
    candidate_pool: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    labels = build_allowed_labels(gold, audit_gold, str(scenario["include_extra"]))
    original_cases = set(gold)
    extra_cases = set(labels) - original_cases
    exclude_cases = set(np.asarray(case_ids, dtype=object)[fold_ids == heldout_fold]) if heldout_fold is not None else set()
    allowed_cases = set(labels) - exclude_cases

    stats_by_route: dict[str, dict[str, dict[str, Any]]] = {}
    for route in TRUNK_ROUTES:
        stats_by_route[route] = {
            cid: standard_stats(cid, raw_routes[route], labels, allowed_cases, exclude_cases, route) for cid in case_ids
        }

    mer_labels = [merlin_row_label(row, gold, {cid: labels[cid] for cid in extra_cases}) for row in mer["rows"]]
    mer_allowed_cases = set(labels)
    stats_by_route["merlinroi_bal"] = build_embedding_stats_for_queries(
        "merlinroi_bal",
        case_ids,
        mer["query_index_by_case"],
        mer["embeddings"],
        mer["embeddings"],
        mer["rows"],
        mer_labels,
        mer_allowed_cases,
        exclude_cases,
        allow_augmented_keys=bool(scenario["allow_augmented_keys"]),
        batch_size=batch_size,
        candidate_pool=candidate_pool,
    )

    arr = examples_to_arrays(case_ids, gold, fold_ids, stats_by_route)
    coverage = {
        "allowed_key_cases": len(allowed_cases),
        "extra_key_cases": len(extra_cases),
        "exclude_cases": len(exclude_cases),
        "route_cases_with_hits": {route: int(np.sum(arr["known"][:, ROUTE_INDEX[route]] > 0)) for route in ROUTES},
    }
    return arr, coverage


def sample_weights_for_scenario(arr: dict[str, np.ndarray], scenario: dict[str, Any]) -> np.ndarray:
    y = arr["gold_present"]
    return np.where(y, float(scenario["pos_weight"]), float(scenario["neg_weight"])).astype(np.float64)


def selection_key(metric: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    return (
        -float(metric["fp"]),
        float(metric["specificity"]),
        float(metric["precision"]),
        float(metric["f1"]),
        float(metric["recall"]),
        -float(metric["fn"]),
    )


def full_best_key(metric: dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        float(metric["f1"]),
        float(metric["balanced_accuracy"]),
        float(metric["specificity"]),
        float(metric["precision"]),
        -float(metric["fp"]),
    )


def update_best(
    best: dict[tuple[str, str], dict[str, Any]],
    scope: str,
    objective: str,
    cfg: dict[str, Any],
    metric: dict[str, float],
    pred: np.ndarray,
) -> None:
    key = (scope, objective)
    current = best.get(key)
    if current is None or selection_key(metric) > selection_key(current["metric"]):
        best[key] = {"scope": scope, "objective": objective, "cfg": cfg, "metric": metric, "pred": pred}


def select_rule_configs_for_fold(
    arr: dict[str, np.ndarray],
    configs: list[dict[str, Any]],
    train_mask: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[tuple[str, str], dict[str, Any]]:
    y_train = arr["gold_present"][train_mask]
    w_train = sample_weight[train_mask]
    best: dict[tuple[str, str], dict[str, Any]] = {}
    family_names = ["global"] + sorted({cfg["kind"] for cfg in configs})
    for cfg in configs:
        pred, _ = predict_config(arr, cfg)
        pred_train = pred[train_mask]
        metric = metric_from_bool(y_train, pred_train, w_train)
        scopes = ["global", cfg["kind"]]
        for floor in RECALL_FLOORS:
            if metric["recall"] >= floor:
                objective = f"min_fp_recall_ge_{floor:.2f}".replace(".", "p")
                for scope in scopes:
                    update_best(best, scope, objective, cfg, metric, pred)
        for scope in scopes:
            objective = "best_f1"
            current = best.get((scope, objective))
            if current is None or full_best_key(metric) > full_best_key(current["metric"]):
                best[(scope, objective)] = {"scope": scope, "objective": objective, "cfg": cfg, "metric": metric, "pred": pred}
    # Ensure absent family rows are explicit in the output summary.
    for family in family_names:
        _ = family
    return best


def precompute_rule_predictions(arr: dict[str, np.ndarray], configs: list[dict[str, Any]]) -> np.ndarray:
    preds = np.empty((len(configs), arr["gold_present"].shape[0]), dtype=bool)
    for idx, cfg in enumerate(configs):
        pred, _ = predict_config(arr, cfg)
        preds[idx] = pred
    return preds


def select_rule_configs_for_fold_cached(
    arr: dict[str, np.ndarray],
    configs: list[dict[str, Any]],
    pred_matrix: np.ndarray,
    train_mask: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[tuple[str, str], dict[str, Any]]:
    y_train = arr["gold_present"][train_mask]
    w_train = sample_weight[train_mask]
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for cfg_idx, cfg in enumerate(configs):
        pred = pred_matrix[cfg_idx]
        metric = metric_from_bool(y_train, pred[train_mask], w_train)
        scopes = ["global", cfg["kind"]]
        for floor in RECALL_FLOORS:
            if metric["recall"] >= floor:
                objective = f"min_fp_recall_ge_{floor:.2f}".replace(".", "p")
                for scope in scopes:
                    update_best(best, scope, objective, cfg, metric, pred)
        for scope in scopes:
            objective = "best_f1"
            current = best.get((scope, objective))
            if current is None or full_best_key(metric) > full_best_key(current["metric"]):
                best[(scope, objective)] = {"scope": scope, "objective": objective, "cfg": cfg, "metric": metric, "pred": pred}
    return best


def evaluate_selected_on_mask(
    selected: dict[tuple[str, str], dict[str, Any]],
    arr: dict[str, np.ndarray],
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = arr["gold_present"][mask]
    for (_scope, _objective), item in selected.items():
        pred = item["pred"][mask]
        metric = metric_from_bool(y, pred)
        cfg = item["cfg"]
        rows.append(
            {
                "selection_scope": item["scope"],
                "objective": item["objective"],
                "config_id": cfg["config_id"],
                "kind": cfg["kind"],
                "preset": cfg.get("preset", ""),
                "config_json": config_json(cfg),
                **metric_prefix("eval", metric),
                **metric_prefix("select", item["metric"]),
            }
        )
    return rows


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def feature_matrix(arr: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    names: list[str] = []
    for route, ridx in ROUTE_INDEX.items():
        for key, values in [
            ("margin", arr["margins"][:, ridx].astype(np.float64)),
            ("known", arr["known"][:, ridx].astype(np.float64)),
            ("present_n", arr["present_n"][:, ridx].astype(np.float64)),
            ("absent_n", arr["absent_n"][:, ridx].astype(np.float64)),
            ("top1_present", arr["top1_present"][:, ridx].astype(np.float64)),
            ("top1_absent", arr["top1_absent"][:, ridx].astype(np.float64)),
        ]:
            features.append(values)
            names.append(f"{route}_{key}")
    trunk_score = arr["margins"][:, ROUTE_INDEX["fullct"]] + 1.5 * arr["margins"][:, ROUTE_INDEX["rsuper"]] + 0.75 * arr["margins"][:, ROUTE_INDEX["radgpt"]]
    all_routes_score = trunk_score + 0.25 * arr["margins"][:, ROUTE_INDEX["merlinroi_bal"]]
    features.extend(
        [
            trunk_score.astype(np.float64),
            all_routes_score.astype(np.float64),
            np.sum(arr["margins"][:, route_indices(TRUNK_ROUTES)] > 0, axis=1).astype(np.float64),
            np.sum(arr["margins"] > 0, axis=1).astype(np.float64),
        ]
    )
    names.extend(["trunk_weighted_score", "all_routes_weighted_score", "trunk_positive_routes", "all_positive_routes"])
    return np.stack(features, axis=1), names


def train_logreg(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, l2: float, lr: float, max_iter: int) -> dict[str, Any]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = (x - mean) / std
    xb = np.concatenate([np.ones((xs.shape[0], 1), dtype=np.float64), xs], axis=1)
    w = np.zeros(xb.shape[1], dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    sw = sample_weight.astype(np.float64)
    sw = sw / max(float(np.mean(sw)), 1e-12)
    reg_mask = np.ones_like(w)
    reg_mask[0] = 0.0
    for step in range(1, max_iter + 1):
        p = stable_sigmoid(xb @ w)
        grad = xb.T @ (sw * (p - y)) / xb.shape[0]
        grad += l2 * reg_mask * w
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad * grad)
        w -= lr * (m / (1.0 - 0.9**step)) / (np.sqrt(v / (1.0 - 0.999**step)) + 1e-8)
    return {"w": w, "mean": mean, "std": std}


def predict_logreg(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    xs = (x - model["mean"]) / model["std"]
    xb = np.concatenate([np.ones((xs.shape[0], 1), dtype=np.float64), xs], axis=1)
    return stable_sigmoid(xb @ model["w"])


def learned_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for l2 in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
        for lr in [0.01, 0.03]:
            rows.append({"kind": "learned_linear_gate", "l2": l2, "lr": lr})
    return rows


def select_learned_for_fold(
    arr: dict[str, np.ndarray],
    train_mask: np.ndarray,
    sample_weight: np.ndarray,
    max_iter: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    x, _names = feature_matrix(arr)
    y = arr["gold_present"].astype(np.float64)
    thresholds = [round(0.02 + 0.02 * idx, 4) for idx in range(49)]
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for idx, cfg in enumerate(learned_grid(), start=1):
        model = train_logreg(x[train_mask], y[train_mask], sample_weight[train_mask], float(cfg["l2"]), float(cfg["lr"]), max_iter)
        prob = predict_logreg(model, x)
        for threshold in thresholds:
            pred = prob >= threshold
            metric = metric_from_bool(arr["gold_present"][train_mask], pred[train_mask], sample_weight[train_mask])
            out_cfg = {"config_id": f"L{idx:04d}_{threshold:.2f}", **cfg, "threshold": threshold}
            for floor in RECALL_FLOORS:
                if metric["recall"] >= floor:
                    objective = f"min_fp_recall_ge_{floor:.2f}".replace(".", "p")
                    update_best(best, "learned_linear_gate", objective, out_cfg, metric, pred)
                    update_best(best, "global_learned", objective, out_cfg, metric, pred)
            current = best.get(("learned_linear_gate", "best_f1"))
            if current is None or full_best_key(metric) > full_best_key(current["metric"]):
                best[("learned_linear_gate", "best_f1")] = {
                    "scope": "learned_linear_gate",
                    "objective": "best_f1",
                    "cfg": out_cfg,
                    "metric": metric,
                    "pred": pred,
                }
    return best


def aggregate_oof(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], Counter[str]] = {}
    meta: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["scenario"], row["bank_mode"], row["weight_mode"], row["selection_scope"], row["objective"])
        grouped.setdefault(key, Counter())
        grouped[key]["tp"] += float(row["eval_tp"])
        grouped[key]["tn"] += float(row["eval_tn"])
        grouped[key]["fp"] += float(row["eval_fp"])
        grouped[key]["fn"] += float(row["eval_fn"])
        meta.setdefault(key, row)
    out: list[dict[str, Any]] = []
    for key, counts in grouped.items():
        scenario, bank_mode, weight_mode, scope, objective = key
        tp = counts["tp"]
        tn = counts["tn"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(
            {
                "scenario": scenario,
                "bank_mode": bank_mode,
                "weight_mode": weight_mode,
                "selection_scope": scope,
                "objective": objective,
                "oof_n": tp + tn + fp + fn,
                "oof_tp": tp,
                "oof_tn": tn,
                "oof_fp": fp,
                "oof_fn": fn,
                "oof_accuracy": (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0,
                "oof_precision": precision,
                "oof_recall": recall,
                "oof_specificity": specificity,
                "oof_f1": f1,
                "oof_balanced_accuracy": (recall + specificity) / 2,
            }
        )
    return out


def select_full_train_configs(
    arr: dict[str, np.ndarray],
    configs: list[dict[str, Any]],
    sample_weight: np.ndarray,
    pred_matrix: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if pred_matrix is None:
        selected = select_rule_configs_for_fold(arr, configs, np.ones(arr["gold_present"].shape[0], dtype=bool), sample_weight)
    else:
        selected = select_rule_configs_for_fold_cached(
            arr,
            configs,
            pred_matrix,
            np.ones(arr["gold_present"].shape[0], dtype=bool),
            sample_weight,
        )
    rows = evaluate_selected_on_mask(selected, arr, np.ones(arr["gold_present"].shape[0], dtype=bool))
    return rows


def route_standalone_rows(arr: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = arr["gold_present"]
    for route, idx in ROUTE_INDEX.items():
        for name, pred in {
            "top1": arr["top1_present"][:, idx],
            "top5_majority": arr["present_n"][:, idx] > arr["absent_n"][:, idx],
        }.items():
            rows.append({"route_rule": f"{route}_{name}", **metric_prefix("train", metric_from_bool(y, pred))})
    return rows


def format_count(value: Any) -> str:
    return str(int(round(float(value))))


def build_markdown(summary_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]], path: Path) -> None:
    focus = sorted(
        [row for row in summary_rows if row["objective"] in {"min_fp_recall_ge_0p85", "min_fp_recall_ge_0p80", "best_f1"}],
        key=lambda row: (float(row["oof_fp"]), -float(row["oof_f1"]), -float(row["oof_recall"])),
    )
    lines = [
        "# Train-Only Comprehensive RAG Policy Search",
        "",
        "No evaluation labels or official_merlin_split=val rows are used. OOF validation removes the held-out fold from the retrieval bank.",
        "",
        "## Best OOF Policies",
        "",
        "| scenario | scope | objective | OOF TP/TN/FP/FN | F1/spec/recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in focus[:40]:
        lines.append(
            "| {scenario} | {scope} | {obj} | {tp}/{tn}/{fp}/{fn} | {f1:.4f}/{spec:.4f}/{rec:.4f} |".format(
                scenario=row["scenario"],
                scope=row["selection_scope"],
                obj=row["objective"],
                tp=format_count(row["oof_tp"]),
                tn=format_count(row["oof_tn"]),
                fp=format_count(row["oof_fp"]),
                fn=format_count(row["oof_fn"]),
                f1=float(row["oof_f1"]),
                spec=float(row["oof_specificity"]),
                rec=float(row["oof_recall"]),
            )
        )
    lines.extend(
        [
            "",
            "## Full-Train Selected Policies",
            "",
            "These rows are selected on all original train cases only. They are fixed-policy candidates for later external/test evaluation.",
            "",
            "| scenario | scope | objective | train TP/TN/FP/FN | F1/spec/recall | config_id |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    full_focus = sorted(
        [row for row in full_rows if row["objective"] in {"min_fp_recall_ge_0p85", "min_fp_recall_ge_0p80", "best_f1"}],
        key=lambda row: (float(row["eval_fp"]), -float(row["eval_f1"]), -float(row["eval_recall"])),
    )
    for row in full_focus[:40]:
        lines.append(
            "| {scenario} | {scope} | {obj} | {tp}/{tn}/{fp}/{fn} | {f1:.4f}/{spec:.4f}/{rec:.4f} | {cid} |".format(
                scenario=row["scenario"],
                scope=row["selection_scope"],
                obj=row["objective"],
                tp=format_count(row["eval_tp"]),
                tn=format_count(row["eval_tn"]),
                fp=format_count(row["eval_fp"]),
                fn=format_count(row["eval_fn"]),
                f1=float(row["eval_f1"]),
                spec=float(row["eval_specificity"]),
                rec=float(row["eval_recall"]),
                cid=row["config_id"],
            )
        )
    lines.extend(
        [
            "",
            "## Coverage Snapshot",
            "",
            "| scenario | fold | allowed keys | extra keys | fullct | rsuper | radgpt | merlinroi_bal |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in coverage_rows[:30]:
        lines.append(
            "| {scenario} | {fold} | {keys} | {extra} | {fullct} | {rsuper} | {radgpt} | {mer} |".format(
                scenario=row["scenario"],
                fold=row["fold"],
                keys=row["allowed_key_cases"],
                extra=row["extra_key_cases"],
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

    original_gold = load_original_gold(TRAIN_RECORDS_JSONL)
    audit_gold_all = load_audit_gold(AUXILIARY_TRAIN_JSONL)
    audit_extra = {cid: label for cid, label in audit_gold_all.items() if cid not in original_gold}
    case_ids = sorted(original_gold)
    fold_ids = np.asarray([fold_for_case(cid, args.folds) for cid in case_ids], dtype=np.int16)

    raw_routes = {
        "fullct": load_raw_standard_route(FULLCT_ROUTE),
        "rsuper": load_raw_standard_route(RSUPER_ROUTE),
        "radgpt": load_raw_standard_route(RADGPT_ROUTE),
    }
    mer_emb, mer_rows, mer_meta = load_npz(MERLIN_BAL_KEY_NPZ)
    original_case_set = set(original_gold)
    mer = {"embeddings": mer_emb, "rows": mer_rows, "metadata": mer_meta, "query_index_by_case": choose_query_indices(mer_rows, original_case_set)}

    configs = candidate_configs()
    scenarios = scenario_rows(args.quick)
    print(
        json.dumps(
            {
                "train_cases": len(case_ids),
                "train_counts": dict(Counter(original_gold.values())),
                "extra_audit_counts": dict(Counter(audit_extra.values())),
                "configs": len(configs),
                "scenarios": len(scenarios),
                "skip_learned": bool(args.skip_learned),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    oof_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    standalone_rows: list[dict[str, Any]] = []
    feature_names: list[str] = []

    bank_groups: dict[tuple[str, str, bool], list[dict[str, Any]]] = {}
    for scenario in scenarios:
        bank_groups.setdefault(bank_key(scenario), []).append(scenario)

    for bank_idx, (bank, grouped_scenarios) in enumerate(bank_groups.items(), start=1):
        bank_scenario = grouped_scenarios[0]
        print(
            f"[bank {bank_idx}/{len(bank_groups)}] {bank_scenario['bank_mode']} "
            f"weights={len(grouped_scenarios)}",
            flush=True,
        )
        full_arr, full_coverage = build_arrays_for_fold(
            bank_scenario,
            None,
            case_ids,
            original_gold,
            audit_extra,
            fold_ids,
            raw_routes,
            mer,
            args.embedding_batch_size,
            args.candidate_pool,
        )
        full_pred_matrix = precompute_rule_predictions(full_arr, configs)
        for scenario in grouped_scenarios:
            print(f"  [scenario] {scenario['scenario']}", flush=True)
            full_weight = sample_weights_for_scenario(full_arr, scenario)
            selected_full = select_full_train_configs(full_arr, configs, full_weight, full_pred_matrix)
            for row in selected_full:
                full_rows.append({**scenario, **row})
            for row in route_standalone_rows(full_arr):
                standalone_rows.append({**scenario, **row})
            coverage_rows.append(
                {
                    **scenario,
                    "fold": "full_train",
                    "allowed_key_cases": full_coverage["allowed_key_cases"],
                    "extra_key_cases": full_coverage["extra_key_cases"],
                    **{f"{route}_cases_with_hits": full_coverage["route_cases_with_hits"][route] for route in ROUTES},
                }
            )
        if not feature_names:
            _x, feature_names = feature_matrix(full_arr)

        for fold in range(args.folds):
            arr, coverage = build_arrays_for_fold(
                bank_scenario,
                fold,
                case_ids,
                original_gold,
                audit_extra,
                fold_ids,
                raw_routes,
                mer,
                args.embedding_batch_size,
                args.candidate_pool,
            )
            print(f"  [fold {fold}] feature bank ready", flush=True)
            pred_matrix = precompute_rule_predictions(arr, configs)
            train_mask = arr["fold_ids"] != fold
            heldout_mask = arr["fold_ids"] == fold
            for scenario in grouped_scenarios:
                sample_weight = sample_weights_for_scenario(arr, scenario)
                selected = select_rule_configs_for_fold_cached(arr, configs, pred_matrix, train_mask, sample_weight)
                rows = evaluate_selected_on_mask(selected, arr, heldout_mask)
                for row in rows:
                    oof_rows.append({**scenario, "fold": fold, **row})
                if not args.skip_learned:
                    learned = select_learned_for_fold(arr, train_mask, sample_weight, args.max_learned_iter)
                    learned_rows = evaluate_selected_on_mask(learned, arr, heldout_mask)
                    for row in learned_rows:
                        oof_rows.append({**scenario, "fold": fold, **row})
                coverage_rows.append(
                    {
                        **scenario,
                        "fold": fold,
                        "allowed_key_cases": coverage["allowed_key_cases"],
                        "extra_key_cases": coverage["extra_key_cases"],
                        **{f"{route}_cases_with_hits": coverage["route_cases_with_hits"][route] for route in ROUTES},
                    }
                )

    summary_rows = aggregate_oof(oof_rows)

    write_csv(args.output_dir / "oof_fold_selected_results.csv", oof_rows)
    write_csv(args.output_dir / "oof_summary.csv", summary_rows)
    write_csv(args.output_dir / "full_train_selected_configs.csv", full_rows)
    write_csv(args.output_dir / "route_standalone_train_metrics.csv", standalone_rows)
    write_csv(args.output_dir / "coverage_by_scenario_fold.csv", coverage_rows)
    write_json(args.output_dir / "feature_names.json", {"feature_names": feature_names})

    summary = {
        "leakage_policy": {
            "reads_evaluation_labels": False,
            "uses_official_merlin_split_val": False,
            "selection_uses_train_only": True,
            "oof_heldout_fold_removed_from_retrieval_bank": True,
            "evaluation_evaluation_in_this_run": False,
        },
        "inputs": {
            "training_records_jsonl": str(TRAIN_RECORDS_JSONL),
            "auxiliary_training_records_jsonl": str(AUXILIARY_TRAIN_JSONL),
            "fullct_route": str(FULLCT_ROUTE),
            "rsuper_route": str(RSUPER_ROUTE),
            "radgpt_route": str(RADGPT_ROUTE),
            "merlin_bal_key_npz": str(MERLIN_BAL_KEY_NPZ),
        },
        "counts": {
            "original_train_cases": len(case_ids),
            "original_train_presence": dict(Counter(original_gold.values())),
            "audit_extra_train_cases": len(audit_extra),
            "audit_extra_presence": dict(Counter(audit_extra.values())),
            "candidate_rule_configs": len(configs),
            "scenarios": len(scenarios),
            "folds": args.folds,
            "learned_router_tested": not args.skip_learned,
        },
        "outputs": {
            "oof_fold_selected_results": str(args.output_dir / "oof_fold_selected_results.csv"),
            "oof_summary": str(args.output_dir / "oof_summary.csv"),
            "full_train_selected_configs": str(args.output_dir / "full_train_selected_configs.csv"),
            "route_standalone_train_metrics": str(args.output_dir / "route_standalone_train_metrics.csv"),
            "coverage_by_scenario_fold": str(args.output_dir / "coverage_by_scenario_fold.csv"),
            "results_md": str(args.output_dir / "RESULTS.md"),
        },
        "npz_metadata": {"merlin_bal": mer_meta},
    }
    write_json(args.output_dir / "summary.json", summary)
    build_markdown(summary_rows, full_rows, coverage_rows, args.output_dir / "RESULTS.md")
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "results_md": str(args.output_dir / "RESULTS.md")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
