#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extended analysis utilities for benchmark result inspection.

This script supports deeper comparisons across prediction files, label levels,
news types, unit levels, and controlled-grouping outputs. It is intended for
post-hoc analysis rather than for the primary inference pipeline.
"""

from __future__ import annotations

import re
import ast
import json
import csv
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Iterable, Optional
from collections import defaultdict

Label = str
InstanceId = Tuple[str, str, str, str]  # (guid, unit_level, unit_id, actor)

LEVELS_ALL = ["article", "subevent", "behavior_chain", "story_narrative"]
LEVELS_HIGH = ["article", "behavior_chain", "story_narrative"]
NEWS_TYPES = ["Background_Analysis", "Current", "Feature"]

# ----------------------------
# IO
# ----------------------------
def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def dump_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def resolve_path(p: str, project_root: Path) -> Path:
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (project_root / pp)

# ----------------------------
# Canonicalization (fix "all zeros" due to IID mismatch)
# ----------------------------
def normalize_level(lv: str) -> str:
    lv = str(lv).strip()
    if lv in LEVELS_ALL:
        return lv
    m = {
        "behavior": "behavior_chain",
        "bce": "behavior_chain",
        "story": "story_narrative",
        "sce": "story_narrative",
    }
    return m.get(lv.lower(), lv)

def parse_subevent_id_list_from_unit_id(unit_id: str) -> List[str]:
    s = str(unit_id).strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    nums = re.findall(r"\d+", s)
    return [str(x) for x in nums]

def canonical_unit_id(level: str, unit_id: str) -> str:
    s = "" if unit_id is None else str(unit_id).strip()
    if level == "behavior_chain":
        ids = parse_subevent_id_list_from_unit_id(s)
        # IMPORTANT: keep order as parsed; only normalize formatting (no spaces)
        return "[" + ",".join([f"'{x}'" for x in ids]) + "]"
    return s

def canonical_iid(guid: Any, unit_level: Any, unit_id: Any, actor: Any) -> InstanceId:
    g = str(guid).strip()
    lv = normalize_level(str(unit_level).strip())
    uid = canonical_unit_id(lv, str(unit_id))
    a = str(actor).strip()
    return (g, lv, uid, a)

def _get_label_list(row: Dict[str, Any], keys: List[str]) -> List[str]:
    """
    Return the first non-empty list found among keys.
    If multiple keys exist and are lists, we will UNION them.
    """
    vals: List[str] = []
    for k in keys:
        v = row.get(k, None)
        if isinstance(v, list):
            vals.extend(v)
    return vals

def load_pred_like(path: str) -> Dict[InstanceId, Dict[str, Set[str]]]:
    """
    Supports BOTH formats:
    - gold style:
        aligned: [...]
        contradictory: [...]
    - pred style:
        aligned_with_human_values: [...]
        contradictory_to_human_values: [...]
    """
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")

    out: Dict[InstanceId, Dict[str, Set[str]]] = {}
    for r in data:
        if not isinstance(r, dict):
            continue

        iid = canonical_iid(
            r.get("guid", ""),
            r.get("unit_level", ""),
            r.get("unit_id", ""),
            r.get("actor", ""),
        )

        aligned_raw = _get_label_list(r, ["aligned", "aligned_with_human_values"])
        contra_raw  = _get_label_list(r, ["contradictory", "contradictory_to_human_values"])

        aligned = set(str(x).strip() for x in (aligned_raw or []))
        contra  = set(str(x).strip() for x in (contra_raw or []))

        out[iid] = {"aligned": aligned, "contradictory": contra}

    return out

def subset_pred_by_gold(
    gold: Dict[InstanceId, Dict[str, Set[str]]],
    pred_all: Dict[InstanceId, Dict[str, Set[str]]],
) -> Dict[InstanceId, Dict[str, Set[str]]]:
    # gold keys already canonical; pred keys already canonical
    out: Dict[InstanceId, Dict[str, Set[str]]] = {}
    for iid in gold.keys():
        out[iid] = pred_all.get(iid, {"aligned": set(), "contradictory": set()})
    return out

# ----------------------------
# Eval core (polarity-aware)
# ----------------------------
def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else float(n) / float(d)

def f1_from_pr(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)

def micro_counts(
    gold_sets: Dict[InstanceId, Set[Label]],
    pred_sets: Dict[InstanceId, Set[Label]],
    instances: Iterable[InstanceId],
) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for iid in instances:
        g = gold_sets.get(iid, set())
        p = pred_sets.get(iid, set())
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return tp, fp, fn

def _per_label_counts(
    gold_sets: Dict[InstanceId, Set[Label]],
    pred_sets: Dict[InstanceId, Set[Label]],
    instances: Iterable[InstanceId],
    label_universe: Set[Label],
) -> Dict[Label, Tuple[int, int, int]]:
    counts = {lab: [0, 0, 0] for lab in label_universe}  # tp, fp, fn
    for iid in instances:
        g = gold_sets.get(iid, set())
        p = pred_sets.get(iid, set())
        for lab in (g & p):
            if lab in counts:
                counts[lab][0] += 1
        for lab in (p - g):
            if lab in counts:
                counts[lab][1] += 1
        for lab in (g - p):
            if lab in counts:
                counts[lab][2] += 1
    return {k: (v[0], v[1], v[2]) for k, v in counts.items()}

def macro_prf_gold_supported(
    gold_sets: Dict[InstanceId, Set[Label]],
    pred_sets: Dict[InstanceId, Set[Label]],
    instances: Iterable[InstanceId],
    label_universe: Set[Label],
) -> Tuple[float, float, float]:
    per_lab = _per_label_counts(gold_sets, pred_sets, instances, label_universe)

    gold_present: Set[Label] = set()
    for iid in instances:
        gold_present |= gold_sets.get(iid, set())

    if not gold_present:
        return 0.0, 0.0, 0.0

    ps, rs, f1s = [], [], []
    for lab in gold_present:
        tp, fp, fn = per_lab.get(lab, (0, 0, 0))
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        ps.append(p)
        rs.append(r)
        f1s.append(f1_from_pr(p, r))

    return sum(ps) / len(ps), sum(rs) / len(rs), sum(f1s) / len(f1s)

def split_dir_value(label: str) -> Tuple[str, str]:
    d, v = label.split(":", 1)
    return d, v

def direction_reverse_rate_gold_excl(
    gold_sets: Dict[InstanceId, Set[Label]],
    pred_sets: Dict[InstanceId, Set[Label]],
) -> Tuple[float, int, int, int]:
    instances = set(gold_sets.keys()) | set(pred_sets.keys())
    reverse_total = 0
    denom_total = 0
    ambiguous_total = 0

    for iid in instances:
        g_labels = gold_sets.get(iid, set())
        p_labels = pred_sets.get(iid, set())

        G_a, G_c = set(), set()
        for lab in g_labels:
            d, v = split_dir_value(lab)
            if d == "1":
                G_a.add(v)
            elif d == "0":
                G_c.add(v)

        P_a, P_c = set(), set()
        for lab in p_labels:
            d, v = split_dir_value(lab)
            if d == "1":
                P_a.add(v)
            elif d == "0":
                P_c.add(v)

        ambiguous = G_a & G_c
        ambiguous_total += len(ambiguous)

        G_only_a = G_a - G_c
        G_only_c = G_c - G_a

        flips = (P_a & G_only_c) | (P_c & G_only_a)
        reverse_total += len(flips)
        denom_total += (len(G_only_a) + len(G_only_c))

    return safe_div(reverse_total, denom_total), reverse_total, denom_total, ambiguous_total

def evaluate_sets(
    gold_sets: Dict[InstanceId, Set[Label]],
    pred_sets: Dict[InstanceId, Set[Label]],
    label_universe: Set[Label],
) -> Dict[str, Any]:
    instances = set(gold_sets.keys()) | set(pred_sets.keys())
    tp, fp, fn = micro_counts(gold_sets, pred_sets, instances)
    micro_p = safe_div(tp, tp + fp)
    micro_r = safe_div(tp, tp + fn)
    micro_f1 = f1_from_pr(micro_p, micro_r)

    macro_p, macro_r, macro_f1 = macro_prf_gold_supported(
        gold_sets, pred_sets, instances, label_universe
    )

    drr, rev_cnt, denom_cnt, amb_cnt = direction_reverse_rate_gold_excl(gold_sets, pred_sets)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "support_instances": len(instances),
        "support_gold_labels": int(sum(len(v) for v in gold_sets.values())),
        "support_pred_labels": int(sum(len(v) for v in pred_sets.values())),
        "drr": drr,
        "drr_reverse_count": rev_cnt,
        "drr_denom_gold_excl": denom_cnt,
        "drr_ambiguous_gold_value_count": amb_cnt,
    }

def build_flat_sets(
    data: Dict[InstanceId, Dict[str, Set[str]]],
) -> Dict[InstanceId, Set[str]]:
    out: Dict[InstanceId, Set[str]] = {}
    for iid, d in data.items():
        s: Set[str] = set()
        for lab in d.get("aligned", set()):
            s.add(f"1:{lab}")
        for lab in d.get("contradictory", set()):
            s.add(f"0:{lab}")
        out[iid] = s
    return out

def build_universe(max_id: int) -> Set[str]:
    u = set()
    for v in range(max_id):
        u.add(f"1:{v}")
        u.add(f"0:{v}")
    return u

def eval_report_one_levelspace(
    gold: Dict[InstanceId, Dict[str, Set[str]]],
    pred: Dict[InstanceId, Dict[str, Set[str]]],
    max_label_id: int,
    per_levels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    per_levels = per_levels or LEVELS_ALL
    universe = build_universe(max_label_id)
    gold_flat = build_flat_sets(gold)
    pred_flat = build_flat_sets(pred)

    overall = evaluate_sets(gold_flat, pred_flat, universe)
    per_level: Dict[str, Any] = {}
    for lv in per_levels:
        g2 = {iid: labs for iid, labs in gold_flat.items() if iid[1] == lv}
        p2 = {iid: labs for iid, labs in pred_flat.items() if iid[1] == lv}
        per_level[lv] = evaluate_sets(g2, p2, universe)
    return {"overall": overall, "per_level": per_level}

# ----------------------------
# L1 -> L2 mapping
# ----------------------------
def load_l1_to_l2_map(path: str) -> Dict[str, str]:
    m = load_json(path)
    if not isinstance(m, dict):
        raise ValueError(f"l1tol2 mapping must be dict, got {type(m)}")
    return {str(k): str(v) for k, v in m.items()}

def map_labels_l1_to_l2(
    data: Dict[InstanceId, Dict[str, Set[str]]],
    l1_to_l2: Dict[str, str],
) -> Dict[InstanceId, Dict[str, Set[str]]]:
    out: Dict[InstanceId, Dict[str, Set[str]]] = {}
    for iid, d in data.items():
        a2 = set()
        c2 = set()
        for x in d.get("aligned", set()):
            if x in l1_to_l2:
                a2.add(l1_to_l2[x])
        for x in d.get("contradictory", set()):
            if x in l1_to_l2:
                c2.add(l1_to_l2[x])
        out[iid] = {"aligned": a2, "contradictory": c2}
    return out

def get_l2_size(l1_to_l2: Dict[str, str]) -> int:
    vals = [int(v) for v in l1_to_l2.values()]
    return (max(vals) + 1) if vals else 0

# ----------------------------
# Value Consistency (works for both L1 and mapped L2)
# ----------------------------
def get_related_subevent_ids_for_story_from_gold_row(gold_row: Dict[str, Any]) -> List[str]:
    keys = [
        "related_subevent_ids", "related_ids", "subevent_ids", "subevents",
        "related_subevents", "related_subevent_id_list"
    ]
    for k in keys:
        if k in gold_row:
            v = gold_row[k]
            if isinstance(v, list):
                return [str(x) for x in v]
            if isinstance(v, str):
                lst = parse_subevent_id_list_from_unit_id(v)
                if lst:
                    return lst
    return []

def labelset_dir_value(d: Dict[str, Set[str]]) -> Set[str]:
    s: Set[str] = set()
    for x in d.get("aligned", set()):
        s.add(f"1:{x}")
    for x in d.get("contradictory", set()):
        s.add(f"0:{x}")
    return s

def prf(set_gold: Set[str], set_pred: Set[str]) -> Tuple[float, float, float]:
    tp = len(set_gold & set_pred)
    fp = len(set_pred - set_gold)
    fn = len(set_gold - set_pred)
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    return p, r, f1_from_pr(p, r)

def value_consistency(
    dataset: Dict[InstanceId, Dict[str, Set[str]]],
    gold_rows_raw: Optional[List[Dict[str, Any]]] = None,
    levels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    levels = levels or LEVELS_HIGH

    sub_by_ga: Dict[Tuple[str, str], Dict[str, Dict[str, Set[str]]]] = defaultdict(dict)
    for iid, labs in dataset.items():
        guid, lv, uid, actor = iid
        if lv == "subevent":
            sub_by_ga[(guid, actor)][str(uid)] = labs

    raw_by_iid: Dict[InstanceId, Dict[str, Any]] = {}
    if gold_rows_raw:
        for r in gold_rows_raw:
            if isinstance(r, dict):
                iid = canonical_iid(r.get("guid",""), r.get("unit_level",""), r.get("unit_id",""), r.get("actor",""))
                raw_by_iid[iid] = r

    def related_union(iid: InstanceId) -> Set[str]:
        guid, lv, uid, actor = iid
        submap = sub_by_ga.get((guid, actor), {})
        rel_ids: List[str] = []
        if lv == "article":
            rel_ids = list(submap.keys())
        elif lv == "behavior_chain":
            rel_ids = parse_subevent_id_list_from_unit_id(uid)
        elif lv == "story_narrative":
            rrow = raw_by_iid.get(iid)
            if rrow:
                rel_ids = get_related_subevent_ids_for_story_from_gold_row(rrow)
            if not rel_ids:
                rel_ids = list(submap.keys())
        u: Set[str] = set()
        for sid in rel_ids:
            labs = submap.get(str(sid))
            if labs:
                u |= labelset_dir_value(labs)
        return u

    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        return safe_div(len(a & b), len(a | b))

    per_level: Dict[str, Any] = {}
    all_ps, all_rs, all_fs, all_js = [], [], [], []

    for lv in levels:
        ps, rs, fs, js = [], [], [], []
        n = 0
        n_rel = 0
        for iid, labs in dataset.items():
            if iid[1] != lv:
                continue
            n += 1
            high = labelset_dir_value(labs)
            low_union = related_union(iid)
            if low_union:
                n_rel += 1
            p, r, f = prf(low_union, high)
            ps.append(p); rs.append(r); fs.append(f); js.append(jaccard(low_union, high))
        per_level[lv] = {
            "mean_precision": float(sum(ps) / len(ps)) if ps else 0.0,
            "mean_recall": float(sum(rs) / len(rs)) if rs else 0.0,
            "mean_f1": float(sum(fs) / len(fs)) if fs else 0.0,
            "mean_jaccard": float(sum(js) / len(js)) if js else 0.0,
            "n_instances": n,
            "n_with_related": n_rel,
        }
        all_ps += ps; all_rs += rs; all_fs += fs; all_js += js

    overall = {
        "mean_precision": float(sum(all_ps) / len(all_ps)) if all_ps else 0.0,
        "mean_recall": float(sum(all_rs) / len(all_rs)) if all_rs else 0.0,
        "mean_f1": float(sum(all_fs) / len(all_fs)) if all_fs else 0.0,
        "mean_jaccard": float(sum(all_js) / len(all_js)) if all_js else 0.0,
        "n_instances": int(sum(per_level[lv]["n_instances"] for lv in per_level)),
        "n_with_related": int(sum(per_level[lv]["n_with_related"] for lv in per_level)),
    }
    return {"overall": overall, "per_level": per_level}

# ----------------------------
# Aggregation (Eval-2)
# ----------------------------
def build_aggregated_pred_from_subevents(
    gold_instances: Dict[InstanceId, Dict[str, Set[str]]],
    pred_all: Dict[InstanceId, Dict[str, Set[str]]],
    gold_rows_raw: Optional[List[Dict[str, Any]]] = None,
) -> Dict[InstanceId, Dict[str, Set[str]]]:
    sub_by_ga: Dict[Tuple[str, str], Dict[str, Dict[str, Set[str]]]] = defaultdict(dict)
    for iid, labs in pred_all.items():
        guid, lv, uid, actor = iid
        if lv == "subevent":
            sub_by_ga[(guid, actor)][str(uid)] = labs

    raw_by_iid: Dict[InstanceId, Dict[str, Any]] = {}
    if gold_rows_raw:
        for r in gold_rows_raw:
            if isinstance(r, dict):
                iid = canonical_iid(r.get("guid",""), r.get("unit_level",""), r.get("unit_id",""), r.get("actor",""))
                raw_by_iid[iid] = r

    def union_subevents(guid: str, actor: str, sids: List[str]) -> Dict[str, Set[str]]:
        a: Set[str] = set()
        c: Set[str] = set()
        submap = sub_by_ga.get((guid, actor), {})
        for sid in sids:
            labs = submap.get(str(sid))
            if not labs:
                continue
            a |= labs.get("aligned", set())
            c |= labs.get("contradictory", set())
        return {"aligned": set(a), "contradictory": set(c)}

    out: Dict[InstanceId, Dict[str, Set[str]]] = {}
    for iid in gold_instances.keys():
        guid, lv, uid, actor = iid
        if lv == "subevent":
            out[iid] = pred_all.get(iid, {"aligned": set(), "contradictory": set()})
            continue

        submap = sub_by_ga.get((guid, actor), {})
        if lv == "article":
            out[iid] = union_subevents(guid, actor, list(submap.keys()))
        elif lv == "behavior_chain":
            out[iid] = union_subevents(guid, actor, parse_subevent_id_list_from_unit_id(uid))
        elif lv == "story_narrative":
            sids: List[str] = []
            rrow = raw_by_iid.get(iid)
            if rrow:
                sids = get_related_subevent_ids_for_story_from_gold_row(rrow)
            if not sids:
                sids = list(submap.keys())
            out[iid] = union_subevents(guid, actor, sids)
        else:
            out[iid] = pred_all.get(iid, {"aligned": set(), "contradictory": set()})
    return out

# ----------------------------
# Model pred discovery
# ----------------------------
def list_model_pred_files(full_pred_dir: str) -> List[str]:
    p = Path(full_pred_dir)
    if not p.exists():
        raise FileNotFoundError(full_pred_dir)
    files = []
    for sub in sorted(p.iterdir()):
        if sub.is_dir():
            f = sub / "test_pred.json"
            if f.exists():
                files.append(str(f))
    return files

def guess_model_hint_from_run_dir(run_dir_name: str) -> str:
    n = run_dir_name.lower()
    if "llama" in n:
        return "Llama"
    if "ministral" in n:
        return "Ministral"
    if "phi" in n:
        return "Phi"
    if "qwen" in n:
        return "Qwen"
    return run_dir_name.split("_")[0]

def find_cgt_pred_for_model(cgt_pred_dir: str, model_hint: str) -> Optional[str]:
    p = Path(cgt_pred_dir)
    if not p.exists():
        return None
    cand = []
    for sub in p.iterdir():
        if sub.is_dir() and (sub / "test_pred.json").exists():
            name = sub.name.lower()
            if model_hint.lower() in name:
                cand.append(str(sub / "test_pred.json"))
    return sorted(cand)[-1] if cand else None

# ----------------------------
# CSV helpers
# ----------------------------
def flatten_report_for_csv(prefix: str, rep: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    def put(k: str, v: Any):
        row[f"{prefix}.{k}"] = v
    for k, v in rep.get("overall", {}).items():
        put(f"overall.{k}", v)
    for lv, d in rep.get("per_level", {}).items():
        for k, v in d.items():
            put(f"{lv}.{k}", v)
    return row

def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ----------------------------
# Dual-space report: Level1 + Level2_from_Level1
# ----------------------------
def eval_dual_space(
    gold: Dict[InstanceId, Dict[str, Set[str]]],
    pred: Dict[InstanceId, Dict[str, Set[str]]],
    l1_to_l2: Dict[str, str],
    l2_size: int,
    per_levels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # Level-1
    rep_l1 = eval_report_one_levelspace(gold, pred, max_label_id=54, per_levels=per_levels)

    # Level-2-from-Level-1
    gold_l2 = map_labels_l1_to_l2(gold, l1_to_l2)
    pred_l2 = map_labels_l1_to_l2(pred, l1_to_l2)
    rep_l2 = eval_report_one_levelspace(gold_l2, pred_l2, max_label_id=l2_size, per_levels=per_levels)

    return {"level1": rep_l1, "level2_from_level1": rep_l2}

# ----------------------------
# Eval-1
# ----------------------------
def run_eval1(gold_path: str, full_pred_dir: str, out_dir: str, l1_to_l2: Dict[str,str], l2_size: int) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gold_rows_raw = load_json(gold_path)
    gold = load_pred_like(gold_path)

    # gold consistency baselines (both spaces)
    gold_cons_l1 = value_consistency(gold, gold_rows_raw=gold_rows_raw, levels=LEVELS_HIGH)
    gold_cons_l2 = value_consistency(map_labels_l1_to_l2(gold, l1_to_l2), gold_rows_raw=gold_rows_raw, levels=LEVELS_HIGH)
    dump_json({"level1": gold_cons_l1, "level2_from_level1": gold_cons_l2}, str(Path(out_dir)/"gold_value_consistency.json"))

    csv_rows: List[Dict[str, Any]] = []
    for pred_file in list_model_pred_files(full_pred_dir):
        run_dir = Path(pred_file).parent.name
        pred_all = load_pred_like(pred_file)
        pred_sub = subset_pred_by_gold(gold, pred_all)

        rep = eval_dual_space(gold, pred_sub, l1_to_l2, l2_size, per_levels=LEVELS_ALL)

        # consistency on preds (both)
        pred_cons_l1 = value_consistency(pred_sub, gold_rows_raw=gold_rows_raw, levels=LEVELS_HIGH)
        pred_cons_l2 = value_consistency(map_labels_l1_to_l2(pred_sub, l1_to_l2), gold_rows_raw=gold_rows_raw, levels=LEVELS_HIGH)
        rep["value_consistency"] = {"level1": pred_cons_l1, "level2_from_level1": pred_cons_l2}

        dump_json(rep, str(Path(out_dir)/f"{run_dir}.json"))

        row = {"model_run": run_dir, "pred_file": pred_file}
        row.update(flatten_report_for_csv("eval1.level1", rep["level1"]))
        row.update(flatten_report_for_csv("eval1.level2", rep["level2_from_level1"]))
        row["eval1.consistency.level1.mean_f1"] = pred_cons_l1["overall"]["mean_f1"]
        row["eval1.consistency.level2.mean_f1"] = pred_cons_l2["overall"]["mean_f1"]
        csv_rows.append(row)

    write_csv(csv_rows, str(Path(out_dir)/"summary.csv"))

# ----------------------------
# Eval-2
# ----------------------------
def run_eval2(gold_path: str, full_pred_dir: str, out_dir: str, l1_to_l2: Dict[str,str], l2_size: int) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gold_rows_raw = load_json(gold_path)
    gold = load_pred_like(gold_path)

    levels_main = ["article", "behavior_chain", "story_narrative"]
    gold_main = {iid: v for iid, v in gold.items() if iid[1] in levels_main}

    csv_rows: List[Dict[str, Any]] = []
    for pred_file in list_model_pred_files(full_pred_dir):
        run_dir = Path(pred_file).parent.name
        pred_all = load_pred_like(pred_file)

        pred_direct = subset_pred_by_gold(gold, pred_all)
        pred_agg = build_aggregated_pred_from_subevents(gold, pred_all, gold_rows_raw=gold_rows_raw)

        pred_direct_main = {iid: v for iid, v in pred_direct.items() if iid[1] in levels_main}
        pred_agg_main = {iid: v for iid, v in pred_agg.items() if iid[1] in levels_main}

        rep_A = eval_dual_space(gold_main, pred_direct_main, l1_to_l2, l2_size, per_levels=levels_main)
        rep_B = eval_dual_space(gold_main, pred_agg_main, l1_to_l2, l2_size, per_levels=levels_main)

        rep = {"A_direct": rep_A, "B_from_subevents": rep_B}
        dump_json(rep, str(Path(out_dir)/f"{run_dir}.json"))

        row = {"model_run": run_dir, "pred_file": pred_file}
        row.update(flatten_report_for_csv("eval2.A.level1", rep_A["level1"]))
        row.update(flatten_report_for_csv("eval2.A.level2", rep_A["level2_from_level1"]))
        row.update(flatten_report_for_csv("eval2.B.level1", rep_B["level1"]))
        row.update(flatten_report_for_csv("eval2.B.level2", rep_B["level2_from_level1"]))
        csv_rows.append(row)

    write_csv(csv_rows, str(Path(out_dir)/"summary.csv"))

# ----------------------------
# Eval-3
# ----------------------------
def run_eval3(gold_sampled_path: str, gold_cgt_path: str, full_pred_dir: str, cgt_pred_dir: str, out_dir: str, l1_to_l2: Dict[str,str], l2_size: int) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gold = load_pred_like(gold_cgt_path)
    levels = ["behavior_chain", "story_narrative"]
    gold_main = {iid: v for iid, v in gold.items() if iid[1] in levels}

    # =======
    # gold_rows_raw = load_json("/tmp/echv_experiment/dataset/deep_analysis/test/gold_sampled.json")
    gold_full = load_pred_like(gold_sampled_path)
    # ============================

    csv_rows: List[Dict[str, Any]] = []
    for full_pred_file in list_model_pred_files(full_pred_dir):
        run_dir = Path(full_pred_file).parent.name
        model_hint = guess_model_hint_from_run_dir(run_dir)

        pred_full_all = load_pred_like(full_pred_file)
        pred_full_sub = subset_pred_by_gold(gold, pred_full_all)

        # =========================
        gold_full_sub = subset_pred_by_gold(gold, gold_full)
        # =========================

        pred_full_main = {iid: v for iid, v in pred_full_sub.items() if iid[1] in levels}
        gold_full_main = {iid: v for iid, v in gold_full_sub.items() if iid[1] in levels}

        # pred_full_sub = subset_pred_by_gold(gold, pred_full_all)
        # rep_A = eval_dual_space(gold_main, pred_full_main, l1_to_l2, l2_size, per_levels=levels)
        rep_A = eval_dual_space(gold_full_main, pred_full_main, l1_to_l2, l2_size, per_levels=levels)

        cgt_pred_file = find_cgt_pred_for_model(cgt_pred_dir, model_hint)
        if cgt_pred_file:
            pred_cgt_all = load_pred_like(cgt_pred_file)
            pred_cgt_sub = subset_pred_by_gold(gold, pred_cgt_all)
            pred_cgt_main = {iid: v for iid, v in pred_cgt_sub.items() if iid[1] in levels}
            rep_B = eval_dual_space(gold_main, pred_cgt_main, l1_to_l2, l2_size, per_levels=levels)
        else:
            rep_B = {"error": f"cgt pred not found for model_hint={model_hint}"}

        rep = {"A_full_pred": rep_A, "B_cgt_pred": rep_B, "model_hint": model_hint, "cgt_pred_file": cgt_pred_file}
        dump_json(rep, str(Path(out_dir)/f"{run_dir}.json"))

        row = {"model_run": run_dir, "model_hint": model_hint, "full_pred_file": full_pred_file, "cgt_pred_file": cgt_pred_file or ""}
        row.update(flatten_report_for_csv("eval3.A.level1", rep_A["level1"]))
        row.update(flatten_report_for_csv("eval3.A.level2", rep_A["level2_from_level1"]))
        if isinstance(rep_B, dict) and "level1" in rep_B and "level2_from_level1" in rep_B:
            row.update(flatten_report_for_csv("eval3.B.level1", rep_B["level1"]))
            row.update(flatten_report_for_csv("eval3.B.level2", rep_B["level2_from_level1"]))
        else:
            row["eval3.B.error"] = rep_B.get("error") if isinstance(rep_B, dict) else "unknown"
        csv_rows.append(row)

    write_csv(csv_rows, str(Path(out_dir)/"summary.csv"))

# ----------------------------
# Eval-4 (news type)
# ----------------------------
def discover_news_type_gold_files(news_type_gold_root: str) -> Dict[str, str]:
    root = Path(news_type_gold_root)
    if not root.exists():
        raise FileNotFoundError(f"news_type_gold_root not found: {news_type_gold_root}")
    out: Dict[str, str] = {}
    missing: List[str] = []
    for nt in NEWS_TYPES:
        p = root / nt / "gold.json"
        if p.exists():
            out[nt] = str(p)
        else:
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing gold.json:\n" + "\n".join(missing))
    return out

def run_eval4_news_type(news_type_gold_root: str, full_pred_dir: str, out_dir: str, l1_to_l2: Dict[str,str], l2_size: int) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gold_map = discover_news_type_gold_files(news_type_gold_root)
    model_pred_files = list_model_pred_files(full_pred_dir)

    csv_rows: List[Dict[str, Any]] = []
    for news_type in NEWS_TYPES:
        gold_path = gold_map[news_type]
        gold = load_pred_like(gold_path)

        nt_out = Path(out_dir) / news_type
        nt_out.mkdir(parents=True, exist_ok=True)

        for pred_file in model_pred_files:
            run_dir = Path(pred_file).parent.name
            pred_all = load_pred_like(pred_file)
            pred_sub = subset_pred_by_gold(gold, pred_all)

            rep = eval_dual_space(gold, pred_sub, l1_to_l2, l2_size, per_levels=LEVELS_ALL)
            dump_json(rep, str(nt_out / f"{run_dir}.json"))

            row = {"news_type": news_type, "gold_path": gold_path, "model_run": run_dir, "pred_file": pred_file}
            row.update(flatten_report_for_csv("eval4.level1", rep["level1"]))
            row.update(flatten_report_for_csv("eval4.level2", rep["level2_from_level1"]))
            csv_rows.append(row)

    write_csv(csv_rows, str(Path(out_dir)/"summary.csv"))

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".", help="project root (contains dataset/ and results/)")
    ap.add_argument("--out_root", type=str, default="/home/iiserver32/Workbench/wang_yao/echv/deep_analysis/G3/",
                    help="output root folder")

    ap.add_argument("--gold_sampled", type=str, default="dataset/deep_analysis/test/gold_sampled.json")
    ap.add_argument("--gold_cgt", type=str, default="dataset/deep_analysis/test/cgt_BCE_SCE_test_gold.json")

    ap.add_argument("--full_pred_dir", type=str, default="results/deep_analysis/G3/full")
    ap.add_argument("--cgt_pred_dir", type=str, default="results/deep_analysis/G3/cgt")

    # IMPORTANT: root is the folder that directly contains Current/ Background_Analysis/ Feature/
    ap.add_argument("--news_type_gold_root", type=str, default="dataset/deep_analysis/test/A1_expand_match_by_feature64")

    # L1->L2 map
    ap.add_argument("--l1_to_l2_map", type=str, default="dataset/hv/l1tol2_id_mapping.json")

    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()

    gold_sampled_path = str(resolve_path(args.gold_sampled, project_root))
    gold_cgt_path = str(resolve_path(args.gold_cgt, project_root))
    full_pred_dir = str(resolve_path(args.full_pred_dir, project_root))
    cgt_pred_dir = str(resolve_path(args.cgt_pred_dir, project_root))
    news_type_gold_root = str(resolve_path(args.news_type_gold_root, project_root))

    out_root = resolve_path(args.out_root, project_root)

    l1_to_l2 = load_l1_to_l2_map(str(resolve_path(args.l1_to_l2_map, project_root)))
    l2_size = get_l2_size(l1_to_l2)

    # Eval-1
    run_eval1(gold_sampled_path, full_pred_dir, str(out_root / "eval1_sampled_full"), l1_to_l2, l2_size)

    # Eval-2
    run_eval2(gold_sampled_path, full_pred_dir, str(out_root / "eval2_aggregation"), l1_to_l2, l2_size)

    # Eval-3
    run_eval3(gold_sampled_path, gold_cgt_path, full_pred_dir, cgt_pred_dir, str(out_root / "eval3_cgt_compare"), l1_to_l2, l2_size)

    # Eval-4
    run_eval4_news_type(news_type_gold_root, full_pred_dir, str(out_root / "eval4_news_type"), l1_to_l2, l2_size)

    print(f"[DONE] Outputs saved to: {out_root}")

if __name__ == "__main__":
    main()
