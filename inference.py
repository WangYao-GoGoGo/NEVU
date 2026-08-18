#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Inference utilities for the main ECHV experiment runner.

This module builds prompts, runs API/local-model generation, parses JSON-like
model outputs, handles long-input map-reduce inference, and normalizes
predictions into the evaluator format.
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import argparse
from google import genai
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set, Optional, Iterable
from collections import defaultdict, Counter
import copy
import numpy as np
from tqdm import tqdm
import sys
from utils import prompt_utils
from utils import llm_utils
from utils import data_utils
from utils import file_utils
import random
from peft import PeftModel
InstanceId = Tuple[str, str, str, str]
# Optional deps for local models / training
LEVEL_MAP = {
    "story_narrative": "story_composite_event",
    "behavior_chain": "behavior_composite_event",
}
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

# LEVEL_CAP = {
#   "article": (24, 16),          # (aligned_cap, contra_cap)
#   "subevent": (8, 6),
#   "behavior_chain": (10, 8),
#   "story_narrative": (10, 8),
# }
import inspect

def apply_chat_template_compat(tokenizer, messages, *, model_name, tokenize=False, add_generation_prompt=True, enable_thinking=False):
    fn = tokenizer.apply_chat_template
    sig = inspect.signature(fn)
    kwargs = dict(tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    if "Qwen" in model_name:
        kwargs["enable_thinking"] = enable_thinking
    return fn(messages, **kwargs)

LEVEL_CAP = {
  "article": (999, 999),          # (aligned_cap, contra_cap)
  "subevent": (999, 999),
  "behavior_chain": (999, 999),
  "story_narrative": (999, 999),
}

def build_chat_str(tokenizer, system_prompt: str, user_prompt: str, model_name: str) -> str:
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    return apply_chat_template_compat(
        tokenizer, messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   #  Qwen3 disables thinking; Llama ignores this argument automatically
        model_name=model_name
    )

def tokenize_chat_str(
    tokenizer,
    chat_str: str,
    max_input_tokens: int,
) -> List[int]:
    # Return only input_ids; its length is prompt_len
    enc = tokenizer(
        chat_str,
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors=None,
    )
    return enc["input_ids"]

@torch.no_grad()
def local_generate_json_batch_from_ids(
    model,
    tokenizer,
    batch_input_ids: List[List[int]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> List[str]:
    """
    batch_input_ids: List of token id lists, already truncated.
    Return: List[str] decoded generations (prompt removed)
    """
    assert isinstance(batch_input_ids, list) and len(batch_input_ids) > 0

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    pad_id = tokenizer.pad_token_id

    # left padding
    lens = [len(x) for x in batch_input_ids]
    max_len = max(lens)
    input_ids = []
    attn = []
    for ids in batch_input_ids:
        pad = max_len - len(ids)
        input_ids.append([pad_id] * pad + ids)
        attn.append([0] * pad + [1] * len(ids))

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)
    attention_mask = torch.tensor(attn, dtype=torch.long, device=model.device)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
    )

    pad_prompt_len = input_ids.shape[1]  # = max_len
    outs = []
    for i in range(gen.shape[0]):
        out_ids = gen[i, pad_prompt_len:]  #  Correct: remove pad plus prompt
        txt = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        outs.append(txt)
    return outs

def chunk_text_by_tokens(tokenizer, text: str, window: int, overlap: int) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return [""]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= window:
        return [text]
    step = max(1, window - overlap)
    chunks = []
    for s in range(0, len(ids), step):
        e = min(len(ids), s + window)
        chunks.append(tokenizer.decode(ids[s:e], skip_special_tokens=True))
        if e >= len(ids):
            break
    return chunks

def reduce_vote(
    chunk_objs: List[Dict[str, Any]],
    *,
    min_votes: int = 2,
    drop_conflicts: bool = True
) -> Dict[str, Any]:
    a_cnt = Counter()
    c_cnt = Counter()

    for obj in chunk_objs:
        a_set, c_set = normalize_pred_labels(obj if obj else {})
        for x in a_set: a_cnt[x] += 1
        for x in c_set: c_cnt[x] += 1

    a_keep = {k for k, v in a_cnt.items() if v >= min_votes}
    c_keep = {k for k, v in c_cnt.items() if v >= min_votes}

    conflicts = a_keep & c_keep
    if conflicts:
        if drop_conflicts:
            a_keep -= conflicts
            c_keep -= conflicts
        else:
            for k in list(conflicts):
                if a_cnt[k] > c_cnt[k]:
                    c_keep.discard(k)
                elif c_cnt[k] > a_cnt[k]:
                    a_keep.discard(k)
                else:
                    a_keep.discard(k)
                    c_keep.discard(k)

    aligned = sorted(a_keep, key=lambda k: (-a_cnt[k], k))
    contra  = sorted(c_keep, key=lambda k: (-c_cnt[k], k))
    print("[reduce_vote] min_votes=", min_votes, "n_chunks=", len(chunk_objs))
    return {"aligned_with_human_values": aligned, "contradictory_to_human_values": contra}

def reduce_adaptive(chunk_objs, drop_conflicts=True):
    n = len(chunk_objs)
    if n == 0:
        return {"aligned_with_human_values": [], "contradictory_to_human_values": []}

    # Important: use a lower threshold for few chunks so different content can still accumulate
    if n <= 2:
        min_votes = 1
    else:
        min_votes = max(2, math.ceil(n * 0.3))

    a_cnt, c_cnt = Counter(), Counter()
    for obj in chunk_objs:
        a, c = normalize_pred_labels(obj if obj else {})
        for x in a: a_cnt[x] += 1
        for x in c: c_cnt[x] += 1

    a_keep = {k for k, v in a_cnt.items() if v >= min_votes}
    c_keep = {k for k, v in c_cnt.items() if v >= min_votes}

    conflicts = a_keep & c_keep
    if conflicts and drop_conflicts:
        a_keep -= conflicts
        c_keep -= conflicts

    aligned = sorted(a_keep, key=lambda k: (-a_cnt[k], k))
    contra  = sorted(c_keep, key=lambda k: (-c_cnt[k], k))
    print("[reduce_vote] min_votes=", min_votes, "n_chunks=", len(chunk_objs))
    return {"aligned_with_human_values": aligned, "contradictory_to_human_values": contra}

def _sort_label_ids(xs: Set[str]) -> List[str]:
    try:
        return sorted(xs, key=lambda x: int(x))
    except ValueError:
        return sorted(xs)

def _stable_unique(xs):
    seen = set()
    out = []
    for x in xs:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def _truncate_obj(obj: Dict[str, Any], topk_a: int, topk_c: int) -> Dict[str, Any]:
    a = obj.get("aligned_with_human_values", []) or []
    c = obj.get("contradictory_to_human_values", []) or []
    if not isinstance(a, list): a = []
    if not isinstance(c, list): c = []
    a = _stable_unique(a)[:topk_a]
    c = _stable_unique(c)[:topk_c]
    return {"aligned_with_human_values": a, "contradictory_to_human_values": c}

def reduce_union_dedup_with_chunk_cap(
    chunk_objs: List[Dict[str, Any]],
    *,
    per_chunk_topk_aligned: int = 999,
    per_chunk_topk_contra: int = 999,
    conflict_mode: str = "prefer_majority",  # "drop" | "prefer_contradictory" | "prefer_aligned" | "prefer_majority"
) -> Dict[str, Any]:
    a_all: Set[str] = set()
    c_all: Set[str] = set()
    a_cnt = Counter()
    c_cnt = Counter()

    def _norm_sort(xs):
        xs = list({str(x).strip() for x in xs if str(x).strip()})
        try:
            xs.sort(key=lambda x: int(x))
        except ValueError:
            xs.sort()
        return xs

    for obj in chunk_objs:
        obj = obj if isinstance(obj, dict) else {}
        a = _stable_unique(obj.get("aligned_with_human_values", []) or [])[:per_chunk_topk_aligned]
        c = _stable_unique(obj.get("contradictory_to_human_values", []) or [])[:per_chunk_topk_contra]

        for x in a:
            a_all.add(x); a_cnt[x] += 1
        for x in c:
            c_all.add(x); c_cnt[x] += 1

    conflicts = a_all & c_all
    if conflicts:
        if conflict_mode == "drop":
            a_all -= conflicts
            c_all -= conflicts
        elif conflict_mode == "prefer_contradictory":
            a_all -= conflicts
        elif conflict_mode == "prefer_aligned":
            c_all -= conflicts
        else:
            # prefer_majority: used only for conflict resolution, not vote filtering
            for k in list(conflicts):
                if a_cnt[k] > c_cnt[k]:
                    c_all.discard(k)
                elif c_cnt[k] > a_cnt[k]:
                    a_all.discard(k)
                else:
                    # On ties, drop contradictory to avoid duplicate polarities
                    # a_all.discard(k)
                    c_all.discard(k)

    def _sort(xs: Set[str]) -> List[str]:
        try:
            return sorted(xs, key=lambda x: int(x))
        except ValueError:
            return sorted(xs)

    return {
        "aligned_with_human_values": _sort(a_all),
        "contradictory_to_human_values": _sort(c_all),
    }

# ----------------------------
# Label universes (IDs as strings for JSON output)
# ----------------------------
L1_IDS = [str(i) for i in range(54)]
L2_IDS = [str(i) for i in range(20)]

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

def str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "t")

def reduce_chunk_predictions(
    preds: List[Dict[str, Any]],
    *,
    min_support: Optional[int] = None,   # None means automatic: ceil(n_chunks * 0.3)
    drop_conflicts: bool = True
) -> Dict[str, List[str]]:
    """
    preds: list of {"aligned_with_human_values":[...], "contradictory_to_human_values":[...]}
    Return final {"aligned_with_human_values": [...], "contradictory_to_human_values": [...]}
    """
    n = len(preds)
    if n == 0:
        return {"aligned_with_human_values": [], "contradictory_to_human_values": []}

    if min_support is None:
        min_support = max(1, math.ceil(n * 0.3))  # 30 percent window support is sufficient

    a = Counter()
    c = Counter()

    for p in preds:
        for x in p.get("aligned_with_human_values", []) or []:
            a[str(x)] += 1
        for x in p.get("contradictory_to_human_values", []) or []:
            c[str(x)] += 1

    # Filter by threshold first
    a_keep = {k for k, v in a.items() if v >= min_support}
    c_keep = {k for k, v in c.items() if v >= min_support}

    # Conflict handling
    conflicts = a_keep & c_keep
    if conflicts:
        if drop_conflicts:
            a_keep -= conflicts
            c_keep -= conflicts
        else:
            # Use majority vote to decide direction; drop on ties
            for k in list(conflicts):
                if a[k] > c[k]:
                    c_keep.discard(k)
                elif c[k] > a[k]:
                    a_keep.discard(k)
                else:
                    a_keep.discard(k)
                    c_keep.discard(k)

    # Sort output by descending count for stability
    aligned = sorted(a_keep, key=lambda k: (-a[k], k))
    contra  = sorted(c_keep, key=lambda k: (-c[k], k))
    return {"aligned_with_human_values": aligned, "contradictory_to_human_values": contra}

def normalize_pred_labels(obj: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    if not isinstance(obj, dict):
        return set(), set()
    a = obj.get("aligned_with_human_values", []) or []
    c = obj.get("contradictory_to_human_values", []) or []
    if not isinstance(a, list): a = []
    if not isinstance(c, list): c = []
    a_set = {str(x).strip() for x in a if str(x).strip()}
    c_set = {str(x).strip() for x in c if str(x).strip()}
    return a_set, c_set


KEY_A = "aligned_with_human_values"
KEY_C = "contradictory_to_human_values"

def extract_json_object(text: Any) -> Optional[Dict[str, Any]]:
    """
    Return a JSON dict ONLY if it contains KEY_A or KEY_C.
    Robust to extra text around JSON.
    """
    if not isinstance(text, (str, bytes)):
        return None
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", errors="ignore")
        except Exception:
            return None

    s = text.strip()
    if not s:
        return None

    # 1) Fast path: whole string is JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and (KEY_A in obj or KEY_C in obj):
            return obj
    except Exception:
        pass

    # 2) Find top-level {...} spans by brace matching (ignore braces inside strings)
    spans = []
    stack = 0
    start = None
    in_str = False
    esc = False

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    spans.append((start, i + 1))
                    start = None

    # 3) Try candidates that contain required keys; pick the “best” one
    best = None
    best_score = -1

    for a, b in spans:
        chunk = s[a:b]
        if (KEY_A not in chunk) and (KEY_C not in chunk):
            continue
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if (KEY_A not in obj) and (KEY_C not in obj):
            continue

        # score: prefer both keys + list types + longer chunk
        score = 0
        score += 2 if KEY_A in obj else 0
        score += 2 if KEY_C in obj else 0
        score += 1 if isinstance(obj.get(KEY_A, []), list) else 0
        score += 1 if isinstance(obj.get(KEY_C, []), list) else 0
        score = score * 10 + len(chunk)

        if score > best_score:
            best_score = score
            best = obj

    return best


# ----------------------------
# G2 Local inference (no training)
# ----------------------------
@torch.no_grad() if torch else (lambda f: f)
def local_generate_json(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    max_input_tokens: Optional[int] = None,   # New
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = apply_chat_template_compat(
        tokenizer, messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        model_name=model_name
    )

    # Use max_input_tokens for truncation instead of tokenizer.model_max_length
    if max_input_tokens is None:
        max_input_tokens = tokenizer.model_max_length

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
    )
    if temperature > 0:
        gen_kwargs["temperature"] = float(temperature)

    gen = model.generate(**gen_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    out = tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True).strip()
    return out

def chunk_evidence_sentences_by_tokens(
    tokenizer,
    evidence_sentences: List[Dict[str, Any]],
    window: int,
    overlap: int,
) -> List[List[Dict[str, Any]]]:
    """
    Split evidence_sentences into chunks by token budget.
    evidence_sentences item format: {"sentence_id": "...", "sentence": "..."} (or "text")
    Returns: list of chunks, each chunk is a list of sentence dicts.
    """
    if not isinstance(evidence_sentences, list) or not evidence_sentences:
        return [[]]

    # normalize: pick sentence text field
    def _get_sent_text(d: Dict[str, Any]) -> str:
        if not isinstance(d, dict):
            return ""
        if isinstance(d.get("sentence"), str):
            return d["sentence"]
        if isinstance(d.get("text"), str):
            return d["text"]
        return ""

    lens = []
    for d in evidence_sentences:
        t = _get_sent_text(d)
        ids = tokenizer(t, add_special_tokens=False)["input_ids"] if t else []
        lens.append(len(ids))

    chunks: List[List[Dict[str, Any]]] = []
    n = len(evidence_sentences)
    start = 0

    while start < n:
        total = 0
        end = start
        while end < n and total + lens[end] <= window:
            total += lens[end]
            end += 1
        if end == start:  # single sentence exceeds window
            end = start + 1

        chunks.append(evidence_sentences[start:end])
        if end >= n:
            break

        # overlap by tokens (approx): move start backward from end until overlap tokens covered
        back_tokens = 0
        new_start = end
        while new_start > start and back_tokens < overlap:
            new_start -= 1
            back_tokens += lens[new_start]
        start = new_start

    return chunks

# ----------------------------
# Inference on a split (for G1/G2/G3)
# ----------------------------
def run_inference(
    group: str,
    provider: str,
    model_name: str,
    api_key: str,
    events_by_guid: Dict[str, Dict[str, Any]],
    gold: Dict[InstanceId, Dict[str, Set[str]]],
    label_space: str,
    hv_label: str,
    hv_label_desc: str,
    prompt_content: str,
    prompt_system: str,
    prompt_variant: str,
    local_model=None,
    local_tokenizer=None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    max_len: int = 4096,
    mr_window: int = 1400,
    mr_overlap: int = 200,
    mr_min_votes: int = 2,
    infer_batch_size: int = 4,
    role1: str = "system",
    role2: str = "user",
    deepseek_base_url: str = "",
    echv_exp_path: str = "",
    g1_use_map_reduce: bool = "",
) -> Dict[InstanceId, Dict[str, Set[str]]]:
    pred: Dict[InstanceId, Dict[str, Set[str]]] = {}
    buf_iids: List[InstanceId] = []
    buf_input_ids: List[List[int]] = []

    if group.upper() == "G1":
        # abc = gold[0:5]
        # ************formal************
        output_file_root_path = echv_exp_path + group + "/output/" + model_name
        output_completed_file_root_path = echv_exp_path + group + "/output_completed/" + model_name
        # ************formal************
        sub_test_data_files = data_utils.split_test_data(gold)
        prompt_cons_file = {}
        for filename, sub_test_data in enumerate(sub_test_data_files):
            sub_test_data_keys = list(sub_test_data.keys())
            # uncompleted_subfiles = []
            user_prompts = []
            for iid in sub_test_data_keys:
                guid = iid[0]

                e = events_by_guid.get(guid)
                if not e:
                    continue
                payload = data_utils.build_input_payload(e, iid, label_space, hv_label, hv_label_desc)
                actors = e["actors"]
                actor_text = data_utils._get_actor_by_id(iid[3], actors)

                completed_file_path = echv_exp_path + group + "/output_completed/" + model_name + "/" + str(
                    filename) + "/completed_results.json"
                file_exist = file_utils.check_path_exist(completed_file_path)

                if file_exist:
                    completed_subfile = file_utils.read_json_file(completed_file_path)
                    completed_keys = data_utils.list_to_dict(completed_subfile)
                else:
                    completed_keys = {}

                unit_level_text = LEVEL_MAP.get(iid[1], iid[1])
                if iid not in completed_keys:
                    user_prompt = prompt_utils.build_prompt_text(payload, prompt_variant, prompt_content, prompt_system, model_name, group)
                    user_prompt_info = {}
                    user_prompt_info["guid"] = guid
                    user_prompt_info["unit_level_text"] = unit_level_text
                    user_prompt_info["unit_level"] = iid[1]
                    user_prompt_info["unit_id"] = iid[2]
                    user_prompt_info["actor_text"] = actor_text
                    user_prompt_info["actor"] = iid[3]
                    user_prompt_info["role1"] = role1
                    user_prompt_info["prompt_system"] = prompt_system
                    user_prompt_info["role2"] = role2
                    user_prompt_info["model_name"] = model_name
                    user_prompt_info["user_prompt"] = user_prompt
                    user_prompt_info["output_file_root_path"] = output_file_root_path
                    user_prompt_info["output_completed_file_root_path"] = output_completed_file_root_path

                    #  The following fields are used for optional MapReduce
                    user_prompt_info["payload"] = payload
                    user_prompt_info["prompt_variant"] = prompt_variant
                    user_prompt_info[
                        "prompt_content_template"] = prompt_content  # template containing <INPUT_VALUE>
                    user_prompt_info["prompt_system"] = prompt_system
                    user_prompt_info["use_map_reduce"] = bool(g1_use_map_reduce)
                    user_prompt_info["max_len"] = int(max_len)
                    user_prompt_info["mr_window"] = int(mr_window)
                    user_prompt_info["mr_overlap"] = int(mr_overlap)
                    user_prompt_info["group"] = group
                    user_prompt_info["model_name"] = model_name
                    user_prompts.append(user_prompt_info)
            if user_prompts:
                prompt_cons_file[str(filename)] = user_prompts

        if model_name.startswith("gemini"):
            client = genai.Client(api_key=api_key)
        elif model_name.startswith("claude"):
            client = llm_utils.anthropic_instantiate(api_key)
        elif model_name.startswith("deepseek"):
            client = llm_utils.open_api_instantiate_deepseek(api_key, deepseek_base_url)
        else:
            client = llm_utils.open_api_instantiate(api_key)

        # prompt_cons_file_new = {}
        # prompt_cons_file_new["0"] = prompt_cons_file["0"][:1]
        # prompt_cons_file_new["1"] = prompt_cons_file["1"][:1]
        # min_thread_count = 20
        min_thread_count = 1
        # min_thread_count = len(prompt_cons_file)
        data_utils.thread_processing(prompt_cons_file, min_thread_count, client)
        return None

    for iid in tqdm(list(gold.keys()), desc=f"Infer {group}:{model_name}"):
        guid = iid[0]
        e = events_by_guid.get(guid)
        if not e:
            continue

        payload = data_utils.build_input_payload(e, iid, label_space, hv_label, hv_label_desc)

        # ---------- Local (G2/G3) ----------
        # -------- decide primary evidence for chunking --------
        unit_text = payload.get("unit_text") or ""
        evidence = payload.get("evidence_sentences") or []
        context = payload.get("context_sentences") or []

        # ---- Build prompt ONCE (no extra prompt_token_len) ----
        user_prompt = prompt_utils.build_prompt_text(payload, prompt_variant, prompt_content, prompt_system, model_name, group)
        chat_str = build_chat_str(local_tokenizer, prompt_system, user_prompt, model_name)

        # 1) First compute the untruncated length
        full_ids = local_tokenizer(chat_str, add_special_tokens=False, truncation=False)["input_ids"]
        overflow = (len(full_ids) > max_len)

        # 2) Then truncate during inference to get the actual input
        input_ids = full_ids[-max_len:] if overflow else full_ids
        p_len = len(full_ids)  # This is the true length, not the truncated length

        unit_level = iid[1]
        cap_a, cap_c = LEVEL_CAP.get(unit_level, (12, 12))

        # If fits -> put into batch buffer (NO single generate here)
        if not overflow:
            buf_iids.append(iid)
            buf_input_ids.append(input_ids)

            if len(buf_iids) >= max(1, infer_batch_size):
                outs = local_generate_json_batch_from_ids(
                    local_model, local_tokenizer,
                    buf_input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                for biid, out_text in zip(buf_iids, outs):
                    unit_level_b = biid[1]
                    cap_a_b, cap_c_b = LEVEL_CAP.get(unit_level_b, (12, 12))

                    obj = extract_json_object(out_text) or {"aligned_with_human_values": [],
                                                            "contradictory_to_human_values": []}
                    obj = _truncate_obj(obj, topk_a=cap_a_b, topk_c=cap_c_b)
                    a_set, c_set = normalize_pred_labels(obj)
                    pred[biid] = {"aligned": a_set, "contradictory": c_set}

                buf_iids, buf_input_ids = [], []

            continue

        # -------- chunk path --------
        chunk_objs: List[Dict[str, Any]] = []

        # Measure content lengths
        unit_text_len = data_utils.calc_token_len(local_tokenizer, unit_text)
        evidence_len = data_utils.calc_sentences_token_len(local_tokenizer, evidence)
        context_len = data_utils.calc_sentences_token_len(local_tokenizer, context)

        # Determine chunking strategy based on content distribution
        RATIO_THRESHOLD = 1.5  # If evidence is 2x longer than unit_text, prefer chunking evidence

        chunk_objs: List[Dict[str, Any]] = []

        # Decision logic:
        # 1. If unit_text is dominant (> evidence * ratio), chunk unit_text
        # 2. If evidence is dominant (> unit_text * ratio), chunk evidence
        # 3. If similar, prefer chunking the one that's not empty, or chunk both

        if unit_text_len > evidence_len * RATIO_THRESHOLD:
        # Case 1: unit_text is non-empty -> chunk unit_text
            chunks = chunk_text_by_tokens(local_tokenizer, unit_text, window=mr_window, overlap=mr_overlap)
            for ch in chunks:
                payload_ch = copy.deepcopy(payload)
                payload_ch["unit_text"] = ch  # chunk main content
                # keep evidence/context as-is (or optionally keep evidence only for grounding)
                user_prompt_ch = prompt_utils.build_prompt_text(payload_ch, prompt_variant, prompt_content,
                                                                prompt_system, model_name, group)
                out_text = local_generate_json(
                    local_model, local_tokenizer,
                    system_prompt=prompt_system,
                    user_prompt=user_prompt_ch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    max_input_tokens=max_len,
                    model_name=model_name,
                )
                obj = extract_json_object(out_text)
                if not obj:
                    obj = {"aligned_with_human_values": [], "contradictory_to_human_values": []}
                chunk_objs.append(obj)

        # Case 2: unit_text is empty (common for article) -> chunk evidence_sentences
        else:
            ev_chunks = chunk_evidence_sentences_by_tokens(
                local_tokenizer, evidence, window=mr_window, overlap=mr_overlap
            )
            for ev in ev_chunks:
                payload_ch = copy.deepcopy(payload)
                payload_ch["evidence_sentences"] = ev
                user_prompt_ch = prompt_utils.build_prompt_text(payload_ch, prompt_variant, prompt_content,
                                                                prompt_system, model_name, group)

                out_text = local_generate_json(
                    local_model, local_tokenizer,
                    system_prompt=prompt_system,
                    user_prompt=user_prompt_ch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    max_input_tokens=max_len,
                    model_name=model_name,
                )
                obj = extract_json_object(out_text)
                if not obj:
                    obj = {"aligned_with_human_values": [], "contradictory_to_human_values": []}
                chunk_objs.append(obj)

        # aggregate (union + dedup + conflict resolve)
        final_obj = reduce_union_dedup_with_chunk_cap(
            chunk_objs=chunk_objs,
            conflict_mode="prefer_majority",
            per_chunk_topk_aligned=cap_a,
            per_chunk_topk_contra=cap_c,
        )
        a_set, c_set = normalize_pred_labels(final_obj)
        pred[iid] = {"aligned": a_set, "contradictory": c_set}
        continue

    # flush remaining buffered samples
    if buf_iids:
        outs = local_generate_json_batch_from_ids(
            local_model, local_tokenizer,
            buf_input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        for biid, out_text in zip(buf_iids, outs):
            unit_level_b = biid[1]
            cap_a_b, cap_c_b = LEVEL_CAP.get(unit_level_b, (12, 12))

            obj = extract_json_object(out_text) or {"aligned_with_human_values": [],
                                                    "contradictory_to_human_values": []}
            obj = _truncate_obj(obj, topk_a=cap_a_b, topk_c=cap_c_b)
            a_set, c_set = normalize_pred_labels(obj)
            pred[biid] = {"aligned": a_set, "contradictory": c_set}

    return pred
