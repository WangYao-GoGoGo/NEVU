#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation utilities for polarity-aware multilabel human-value prediction.

The evaluator represents each instance by ``(guid, unit_level, unit_id, actor)``
and compares aligned/contradictory human-value sets at both level-1 and derived
level-2 label spaces.
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from config import config_unified_value_recognition as config
import sys
from utils import prompt_utils
from pathlib import Path

# Optional deps for local models / training
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None

# transformers + peft for G2/G3
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import get_linear_schedule_with_warmup
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None


# ----------------------------
# Label universes (IDs as strings for JSON output)
# ----------------------------
L1_IDS = [str(i) for i in range(54)]
L2_IDS = [str(i) for i in range(20)]

import random
from typing import Iterable

from peft import PeftModel

# ===== ADD 1) Added in the import section at the top of the file =====
from typing import Optional

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

def str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def map_l1_dict_to_l2(
    data: Dict[InstanceId, Dict[str, Set[str]]],
    l1_to_l2: Dict[str, str],
) -> Dict[InstanceId, Dict[str, Set[str]]]:
    out: Dict[InstanceId, Dict[str, Set[str]]] = {}
    for iid, d in data.items():
        a = set()
        c = set()
        for x in d.get("aligned", set()):
            k = str(x)
            if k in l1_to_l2:
                a.add(str(l1_to_l2[k]))
        for x in d.get("contradictory", set()):
            k = str(x)
            if k in l1_to_l2:
                c.add(str(l1_to_l2[k]))
        out[iid] = {"aligned": a, "contradictory": c}
    return out

def build_eval_views(
    gold: Dict[InstanceId, Dict[str, Set[str]]],
    pred: Dict[InstanceId, Dict[str, Set[str]]],
    label_space: str,
    compute_drr: bool = False,
) -> Dict[str, Any]:
    """
    We evaluate polarity-aware multi-label by flattening into "dir:label" strings.
    Your convention: direction=1 aligned, direction=0 contradictory.
    """
    universe = set()
    base = range(54) if label_space.upper() == "L1" else range(20)
    for v in base:
        universe.add(f"1:{v}")
        universe.add(f"0:{v}")

    gold_flat = {}
    pred_flat = {}

    for iid, d in gold.items():
        s = set()
        for lab in d["aligned"]:
            s.add(f"1:{lab}")
        for lab in d["contradictory"]:
            s.add(f"0:{lab}")
        gold_flat[iid] = s

    for iid, d in pred.items():
        s = set()
        for lab in d["aligned"]:
            s.add(f"1:{lab}")
        for lab in d["contradictory"]:
            s.add(f"0:{lab}")
        pred_flat[iid] = s

    overall = evaluate_sets(gold_flat, pred_flat, universe, compute_drr=compute_drr)

    levels = ["article", "subevent", "behavior_chain", "story_narrative"]
    per_level = {}
    for lv in levels:
        g2 = {iid: labs for iid, labs in gold_flat.items() if iid[1] == lv}
        p2 = {iid: labs for iid, labs in pred_flat.items() if iid[1] == lv}
        per_level[lv] = evaluate_sets(g2, p2, universe, compute_drr=compute_drr)

    return {"overall": overall, "per_level": per_level}

def build_eval_report_l1_and_l2_from_l1(
    gold_l1: Dict[InstanceId, Dict[str, Set[str]]],
    pred_l1: Dict[InstanceId, Dict[str, Set[str]]],
    l1tol2_mapping_path: str,
) -> Dict[str, Any]:
    l1_to_l2 = load_json(l1tol2_mapping_path)
    # L1 with DRR
    l1_report = build_eval_views(gold_l1, pred_l1, label_space="L1", compute_drr=True)

    # L2 derived from L1
    gold_l2 = map_l1_dict_to_l2(gold_l1, l1_to_l2)
    pred_l2 = map_l1_dict_to_l2(pred_l1, l1_to_l2)
    l2_report = build_eval_views(gold_l2, pred_l2, label_space="L2", compute_drr=True)

    return {
        "level1": l1_report,
        "level2_from_level1": l2_report,
    }

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
    """
    Return counts per label: {lab: (tp, fp, fn)}.
    """
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
    """
    Gold-supported macro P/R/F1:
      - Only average over labels that appear at least once in GOLD across these instances.
    """
    per_lab = _per_label_counts(gold_sets, pred_sets, instances, label_universe)

    # which labels are present in gold?
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

    macro_p = sum(ps) / len(ps)
    macro_r = sum(rs) / len(rs)
    macro_f1 = sum(f1s) / len(f1s)
    return macro_p, macro_r, macro_f1

# -------- DRR (Direction Reverse Rate) --------
def split_dir_value(label: str) -> Tuple[str, str]:
    """
    label format: "{dir}:{value_id}"  e.g., "1:42"
    dir: "1" aligned, "0" contradictory   (keep this consistent with the existing flattening logic)
    """
    d, v = label.split(":", 1)
    return d, v

def direction_reverse_rate_gold_excl(
    gold_sets: Dict[InstanceId, Set[Label]],
    pred_sets: Dict[InstanceId, Set[Label]],
) -> Tuple[float, int, int, int]:
    """
    DRR (gold-exclusive):
    - If a value appears as both aligned and contradictory for the same gold instance, mark it ambiguous and exclude it from the denominator
    - Count direction reversals only for values with a unique gold direction:
        gold_only_aligned predicted as contra, or gold_only_contra predicted as aligned
    """
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

    drr = safe_div(reverse_total, denom_total)
    return drr, reverse_total, denom_total, ambiguous_total

def macro_f1_gold_supported(
    gold_sets: Dict[InstanceId, Set[str]],
    pred_sets: Dict[InstanceId, Set[str]],
    instances: Set[InstanceId],
    label_universe: Set[str],
) -> float:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for iid in instances:
        g = gold_sets.get(iid, set())
        p = pred_sets.get(iid, set())
        for lab in (g & p):
            tp[lab] += 1
        for lab in (p - g):
            fp[lab] += 1
        for lab in (g - p):
            fn[lab] += 1

    labels = [lab for lab in label_universe if (tp[lab] + fn[lab]) > 0]  # gold-supported
    if not labels:
        return 0.0

    f1s = []
    for lab in labels:
        p = safe_div(tp[lab], tp[lab] + fp[lab])
        r = safe_div(tp[lab], tp[lab] + fn[lab])
        f1s.append(f1_from_pr(p, r))
    return float(sum(f1s) / len(f1s))

# ----------------------------
# Metrics: micro + gold-supported macro (P/R/F1) + DRR
# ----------------------------
def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else float(n) / float(d)

def f1_from_pr(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)

def evaluate_sets(
    gold_sets: Dict[InstanceId, Set[Label]],
    pred_sets: Dict[InstanceId, Set[Label]],
    label_universe: Set[Label],
    compute_drr: bool = False,
) -> Dict[str, Any]:
    instances = set(gold_sets.keys()) | set(pred_sets.keys())
    tp, fp, fn = micro_counts(gold_sets, pred_sets, instances)
    micro_p = safe_div(tp, tp + fp)
    micro_r = safe_div(tp, tp + fn)
    micro_f1 = f1_from_pr(micro_p, micro_r)

    macro_p, macro_r, macro_f1 = macro_prf_gold_supported(
        gold_sets, pred_sets, instances, label_universe
    )

    out = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "support_instances": len(instances),
        "support_gold_labels": int(sum(len(v) for v in gold_sets.values())),
        "support_pred_labels": int(sum(len(v) for v in pred_sets.values())),
    }

    if compute_drr:
        drr, rev_cnt, denom_cnt, amb_cnt = direction_reverse_rate_gold_excl(gold_sets, pred_sets)
        out.update({
            "drr": drr,
            "drr_reverse_count": rev_cnt,
            "drr_denom_gold_excl": denom_cnt,
            "drr_ambiguous_gold_value_count": amb_cnt,
        })

    return out
