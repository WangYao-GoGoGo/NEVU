#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Ablation-study runner for the ECHV experimental pipeline.

The script evaluates input ablations for the same task format used by the main
runner. Ablation variants remove or simplify selected prompt fields, such as
actor information, evidence sentences, context sentences, or full unit context.
It keeps separate execution paths so ablation results do not modify the main
inference utilities.
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import re
import json
import math
import time
import copy
import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set, Optional, Iterable
from collections import defaultdict
from argparse import Namespace
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm
import sys

from config import config_unified_value_recognition as config
from utils import prompt_utils
import inference as infer
import evaluate as eval
from utils import file_utils
from utils import data_utils

try:
    import nonllm_retrieval as nonllm_infer
except Exception:
    nonllm_infer = None

# Optional deps for local models / training
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    Dataset = object
    DataLoader = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import get_linear_schedule_with_warmup
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    get_linear_schedule_with_warmup = None

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def str2bool(x: Any) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def _count_unique_guids(events_list: List[Dict[str, Any]]) -> int:
    return len({str(e.get("guid", "")).strip() for e in events_list if isinstance(e, dict) and str(e.get("guid", "")).strip()})


def _norm_iid_from_hv_row(r: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Best-effort normalization for rows in hv_rows_formatted.json.
    This matches the tuple format used by data_utils.list_to_gold_dict.
    """
    guid = str(r.get("guid", "")).strip()
    unit_level = str(r.get("unit_level", "")).strip()
    unit_id = str(r.get("unit_id", "")).strip()
    actor = str(r.get("actor", "")).strip()
    return guid, unit_level, unit_id, actor


def _resolve_run_dir_from_resume_path(resume_path: str) -> str:
    """
    resume_path can be:
      - .../<run>/checkpoints/epoch_4/
      - .../<run>/best/
      - .../<run>/checkpoints/
      - .../<run>/
    Return <run>.
    """
    p = os.path.abspath(resume_path).rstrip("/")
    base = os.path.basename(p)
    parent = os.path.basename(os.path.dirname(p))

    if re.match(r"^epoch_\d+$", base) and parent == "checkpoints":
        return os.path.dirname(os.path.dirname(p))
    if base == "best":
        return os.path.dirname(p)
    if base == "checkpoints":
        return os.path.dirname(p)
    return p


def _rng_state_dict() -> Dict[str, Any]:
    if torch is None:
        return {"python": random.getstate(), "numpy": np.random.get_state()}
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state_dict(st: Dict[str, Any]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    if torch is not None and "torch" in st:
        torch.set_rng_state(st["torch"])
        if torch.cuda.is_available() and "cuda" in st:
            torch.cuda.set_rng_state_all(st["cuda"])


def save_checkpoint_lora(
    ckpt_dir: str,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler=None,
    epoch: int = 0,
    global_step: int = 0,
    extra: Optional[dict] = None,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": _rng_state_dict(),
        "extra": extra or {},
    }
    torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))


def load_checkpoint_lora(
    ckpt_dir: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    st_path = os.path.join(ckpt_dir, "trainer_state.pt")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"Missing trainer_state.pt in {ckpt_dir}")

    state = torch.load(st_path, map_location=map_location, weights_only=False)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if "rng" in state and state["rng"] is not None:
        _set_rng_state_dict(state["rng"])
    return state


def dump_sampled_split_files(
    *,
    split_name: str,
    events_list: List[Dict[str, Any]],
    hv_rows_list: List[Dict[str, Any]],
    sampled_gold: Dict[Tuple[str, str, str, str], Dict[str, Set[str]]],
    out_dir: str,
    file_utils,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    keep_iids = set(sampled_gold.keys())
    keep_guids = {iid[0] for iid in keep_iids}

    sampled_events = []
    if isinstance(events_list, list):
        for e in events_list:
            if not isinstance(e, dict):
                continue
            g = str(e.get("guid", "")).strip()
            if g in keep_guids:
                sampled_events.append(e)

    sampled_hv = []
    if isinstance(hv_rows_list, list):
        for r in hv_rows_list:
            if not isinstance(r, dict):
                continue
            iid = _norm_iid_from_hv_row(r)
            if iid in keep_iids:
                sampled_hv.append(r)

    event_path = os.path.join(out_dir, f"{split_name}_event_base.json")
    hv_path = os.path.join(out_dir, f"{split_name}_labels.json")
    file_utils.save_json(event_path, sampled_events)
    file_utils.save_json(hv_path, sampled_hv)

    print(f"[INFO] dumped {split_name}: events={len(sampled_events)}, hv_rows={len(sampled_hv)}")
    print(f"[INFO] -> {event_path}")
    print(f"[INFO] -> {hv_path}")
    return event_path, hv_path


def get_nested(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model_for_infer(
    group: str,
    model_name: str,
    use_auto_map: bool,
    hf_token: Optional[str] = None,
    adapter_dir: str = "",
    load_dir: str = "",
    merge_lora_infer: bool = False,
):
    """
    Priority:
      1) if load_dir: load full model/tokenizer from load_dir
      2) elif group==G3 and adapter_dir: load base model then attach LoRA adapter
      3) else: load base model (G2)
    """
    assert torch is not None and AutoTokenizer is not None and AutoModelForCausalLM is not None, \
        "Need torch+transformers for local inference"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    g = (group or "").upper()

    def _has_tokenizer_files(p: str) -> bool:
        if not p or not os.path.isdir(p):
            return False
        cand = [
            "tokenizer.json", "tokenizer.model",
            "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt",
        ]
        return any(os.path.exists(os.path.join(p, x)) for x in cand)

    def _load_tokenizer(src: str):
        try:
            return AutoTokenizer.from_pretrained(src, token=hf_token, use_fast=True)
        except Exception as e1:
            msg = str(e1)
            if "TokenizersBackend" in msg:
                try:
                    from transformers import MistralCommonBackend
                    return MistralCommonBackend.from_pretrained(src, token=hf_token)
                except Exception as e_mcb:
                    raise RuntimeError(
                        "Ministral tokenizer requires mistral-common backend, but MistralCommonBackend failed.\n"
                        "Please ensure: pip install -U 'mistral-common>=1.8.6' 'transformers>=5.0.0rc0'\n"
                        f"Original AutoTokenizer error: {e1}\n"
                        f"MistralCommonBackend error: {e_mcb}"
                    )
            try:
                return AutoTokenizer.from_pretrained(src, token=hf_token, use_fast=False)
            except Exception:
                return AutoTokenizer.from_pretrained(src, token=hf_token, use_fast=False, trust_remote_code=True)

    def _load_model(src: str):
        common_kwargs = dict(
            token=hf_token,
            torch_dtype=dtype,
            device_map="auto" if use_auto_map else None,
        )
        last_err = None
        try:
            return AutoModelForCausalLM.from_pretrained(src, **common_kwargs)
        except Exception as e_causal:
            last_err = e_causal

        try:
            from transformers import Mistral3ForConditionalGeneration
            return Mistral3ForConditionalGeneration.from_pretrained(src, **common_kwargs)
        except Exception as e_m3:
            last_err = e_m3

        try:
            return AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True, **common_kwargs)
        except Exception as e_trc:
            last_err = e_trc

        try:
            from transformers import Mistral3ForConditionalGeneration
            return Mistral3ForConditionalGeneration.from_pretrained(src, trust_remote_code=True, **common_kwargs)
        except Exception as e_trc2:
            last_err = e_trc2

        raise RuntimeError(f"Failed to load model from {src}. last_err={last_err}")

    tok_candidates = []
    if load_dir:
        tok_candidates.append(load_dir)
    if g == "G3" and adapter_dir and _has_tokenizer_files(adapter_dir):
        tok_candidates.append(adapter_dir)
    tok_candidates.append(model_name)

    tok = None
    last_err = None
    for src in tok_candidates:
        try:
            tok = _load_tokenizer(src)
            break
        except Exception as e:
            last_err = e
    if tok is None:
        raise RuntimeError(f"Failed to load tokenizer. candidates={tok_candidates}. last_err={last_err}")

    tok.truncation_side = "left"
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if load_dir:
        model = _load_model(load_dir)
        if not use_auto_map:
            model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model.eval()
        return model, tok

    base = _load_model(model_name)
    if not use_auto_map:
        base.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    if g == "G3" and adapter_dir:
        assert PeftModel is not None, "peft not installed, cannot load LoRA adapter"
        model = PeftModel.from_pretrained(base, adapter_dir)
        if merge_lora_infer:
            model = model.merge_and_unload()
    else:
        model = base

    model.eval()
    return model, tok


def maybe_save_merged_full_model(model, tok, save_merged_dir: str):
    if not save_merged_dir:
        return
    assert PeftModel is not None, "peft not installed"
    os.makedirs(save_merged_dir, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(save_merged_dir)
    tok.save_pretrained(save_merged_dir)
    print(f"[INFO] merged full model saved to: {save_merged_dir}")


def load_model_and_tokenizer_for_infer(args, token=None, dtype=None, device="cuda"):
    tok = AutoTokenizer.from_pretrained(args.model_name, token=token, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=token,
        torch_dtype=dtype,
        device_map="auto" if args.device_map == "auto" else None,
    )
    if args.device_map == "single":
        base.to(device)

    if args.adapter_dir:
        model = PeftModel.from_pretrained(base, args.adapter_dir)
    else:
        model = base

    model.eval()
    return model, tok


# ---------------------------------------------------------------------
# Gold / SFT examples
# ---------------------------------------------------------------------

InstanceId = Tuple[str, str, str, str]


@dataclass
class SFTExample:
    iid: InstanceId
    prompt: str
    target_json: str


class SFTDataset(Dataset):
    def __init__(self, examples: List[SFTExample], tokenizer, max_len: int = 2048):
        self.examples = examples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_ids = self.tok(ex.prompt, add_special_tokens=False)["input_ids"]
        target_text = "\n" + ex.target_json
        target_ids = self.tok(target_text, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) + len(target_ids) > self.max_len:
            keep_prompt = max(0, self.max_len - len(target_ids))
            prompt_ids = prompt_ids[-keep_prompt:]

        if len(target_ids) > self.max_len:
            target_ids = target_ids[:self.max_len]
            prompt_ids = []

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_collate_fn(tok):
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tok.pad_token_id
        input_ids, attn, labels = [], [], []
        for x in batch:
            l = len(x["input_ids"])
            pad = max_len - l
            input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
            attn.append(torch.cat([x["attention_mask"], torch.zeros((pad,), dtype=torch.long)]))
            labels.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attn, dim=0),
            "labels": torch.stack(labels, dim=0),
        }
    return collate_fn


def gold_to_target_json(g, label_dropout: float = 0.0) -> str:
    a = list(g["aligned"])
    c = list(g["contradictory"])

    if label_dropout > 0:
        def drop(xs):
            if len(xs) <= 1:
                return xs
            kept = [x for x in xs if random.random() > label_dropout]
            return kept if kept else xs[:1]
        a = drop(a)
        c = drop(c)

    obj = {
        "aligned_with_human_values": sorted(a),
        "contradictory_to_human_values": sorted(c),
    }
    return json.dumps(obj, ensure_ascii=False)


def build_sft_examples(
    events_by_guid: Dict[str, Dict[str, Any]],
    gold: Dict[InstanceId, Dict[str, Set[str]]],
    label_space: str,
    hv_label: str,
    hv_label_desc: str,
    prompt_content: str,
    prompt_system: str,
    model_name: str,
    group: str,
    prompt_variant: str = "A",
    label_dropout: float = 0.1,
) -> List[SFTExample]:
    examples = []
    for iid, g in gold.items():
        guid = iid[0]
        e = events_by_guid.get(guid)
        if not e:
            continue
        payload = data_utils.build_input_payload(e, iid, label_space, hv_label, hv_label_desc)
        prompt = prompt_utils.build_prompt_text(payload, prompt_variant, prompt_content, prompt_system, model_name, group)
        target = gold_to_target_json(g, label_dropout=label_dropout)
        examples.append(SFTExample(iid=iid, prompt=prompt, target_json=target))
    return examples


def load_local_model_and_tokenizer(base_model_name: str):
    assert torch is not None and AutoTokenizer is not None and AutoModelForCausalLM is not None

    if "Ministral" in base_model_name or "ministral" in base_model_name:
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
        tok = MistralCommonBackend.from_pretrained(base_model_name)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    else:
        tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    tok.padding_side = "left"
    return model, tok


# ---------------------------------------------------------------------
# LoRA training
# ---------------------------------------------------------------------

def train_lora_sft(
    args: Namespace,
    base_model_name: str,
    output_dir: str,
    train_dataset: SFTDataset,
    dev_gold: Dict[InstanceId, Dict[str, Set[str]]],
    dev_events_by_guid: Dict[str, Dict[str, Any]],
    label_space: str,
    hv_label: str,
    hv_label_desc: str,
    prompt_content: str,
    prompt_system: str,
    prompt_variant: str,
    lr: float = 2e-4,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_new_tokens_eval: int = 256,
    device: str = "cuda",
    max_len_eval: int = 2048,
) -> Tuple[Any, Any]:
    assert torch is not None, "torch not installed"
    assert AutoTokenizer is not None and AutoModelForCausalLM is not None, "transformers not installed"
    assert LoraConfig is not None and get_peft_model is not None, "peft not installed"

    model, tok = load_local_model_and_tokenizer(base_model_name)
    tok.truncation_side = "left"
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    os.makedirs(output_dir, exist_ok=True)
    best_dir = os.path.join(output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    best_score = -1e18
    best_epoch = -1
    if args.resume_lora_dir:
        meta_path = os.path.join(output_dir, "best", "best_meta.json")
        if os.path.exists(meta_path):
            bm = file_utils.load_json(meta_path)
            best_epoch = int(bm.get("best_epoch", -1))
            best_score = float(bm.get("best_score", -1e18))

    epoch_scores = {}

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    if args.resume_lora_dir:
        assert PeftModel is not None, "peft not installed"
        model = PeftModel.from_pretrained(model, args.resume_lora_dir, is_trainable=True)
        print(f"[INFO] Resuming LoRA adapter from: {args.resume_lora_dir}")
    else:
        model = get_peft_model(model, lora_cfg)

    dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_collate_fn(tok))
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps_total = epochs * math.ceil(len(dl) / grad_accum)
    warmup_steps = max(10, total_steps_total // 20)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps_total,
    )

    step = 0
    start_ep = 0
    if args.resume_lora_dir:
        ckpt_state = load_checkpoint_lora(
            ckpt_dir=args.resume_lora_dir,
            model=model,
            optimizer=optim,
            scheduler=sched,
            scaler=None,
            map_location="cpu",
        )
        start_ep = int(ckpt_state.get("epoch", 0))
        step = int(ckpt_state.get("global_step", 0))
        print(f"[INFO]  Loaded trainer_state: epoch={start_ep}, global_step={step}")

    def _save_checkpoint(epoch: int, score: float, reason: str):
        epoch_dir = os.path.join(checkpoints_dir, f"epoch_{epoch}")
        save_checkpoint_lora(
            ckpt_dir=epoch_dir,
            model=model,
            tokenizer=tok,
            optimizer=optim,
            scheduler=sched,
            scaler=None,
            epoch=epoch,
            global_step=step,
            extra={"score": score, "reason": reason, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        )
        meta = {
            "epoch": epoch,
            "score": score,
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        file_utils.save_json(os.path.join(epoch_dir, "checkpoint_meta.json"), meta)
        print(f"[INFO] 💾 Saved checkpoint: {epoch_dir} ({reason}, score={score:.4f})")

    model.zero_grad(set_to_none=True)
    for ep in range(start_ep + 1, epochs + 1):
        print(f"\n{'=' * 60}\nEpoch {ep}/{epochs}\n{'=' * 60}")
        model.train()
        pbar = tqdm(dl, desc=f"Train epoch {ep}")
        for batch_i, batch in enumerate(pbar, start=1):
            batch["input_ids"][batch["attention_mask"] == 0] = tok.pad_token_id
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()

            if batch_i % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        print(f"[INFO] Running dev evaluation for epoch {ep}...")
        model.eval()
        dev_pred = infer.run_inference(
            group="G3",
            provider="",
            model_name=base_model_name + "+LoRA",
            api_key="",
            events_by_guid=dev_events_by_guid,
            gold=dev_gold,
            label_space=label_space,
            hv_label=hv_label,
            hv_label_desc=hv_label_desc,
            prompt_content=prompt_content,
            prompt_system=prompt_system,
            prompt_variant=prompt_variant,
            local_model=model,
            local_tokenizer=tok,
            max_new_tokens=max_new_tokens_eval,
            temperature=args.temperature,
            max_len=args.max_len,
            infer_batch_size=args.infer_batch_size,
        )

        dev_report = {
            "epoch": ep,
            "dev": eval.build_eval_report_l1_and_l2_from_l1(dev_gold, dev_pred, args.l1tol2_mappings),
        }
        file_utils.save_json(args.report_out_dev, dev_report)
        score = get_nested(dev_report["dev"], ["level1", "overall", "micro_f1"], default=float("nan"))
        epoch_scores[ep] = score
        print(f"[INFO] Epoch {ep} score: {score:.4f}")

        should_save = False
        save_reason = ""
        if ep == 1:
            should_save = True
            save_reason = "first epoch"
            if isinstance(score, float) and not math.isnan(score) and (best_epoch == -1 or score > best_score):
                best_score = score
                best_epoch = ep
                save_checkpoint_lora(
                    ckpt_dir=best_dir,
                    model=model,
                    tokenizer=tok,
                    optimizer=optim,
                    scheduler=sched,
                    scaler=None,
                    epoch=ep,
                    global_step=step,
                    extra={"best": True, "score": best_score, "metric": "level1.overall.micro_f1", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                )
                file_utils.save_json(os.path.join(best_dir, "best_meta.json"), {
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "metric": "level1.overall.micro_f1",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"[INFO]  Initialized BEST with epoch 1 (score={best_score:.4f})")
        elif isinstance(score, float) and not math.isnan(score) and score > best_score:
            should_save = True
            save_reason = "new best"
            best_score = score
            best_epoch = ep
            save_checkpoint_lora(
                ckpt_dir=best_dir,
                model=model,
                tokenizer=tok,
                optimizer=optim,
                scheduler=sched,
                scaler=None,
                epoch=ep,
                global_step=step,
                extra={"best": True, "score": best_score, "metric": "level1.overall.micro_f1", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            )
            file_utils.save_json(os.path.join(best_dir, "best_meta.json"), {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "metric": "level1.overall.micro_f1",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"[INFO]  NEW BEST! Saved to {best_dir} (epoch={best_epoch}, score={best_score:.4f})")
        elif ep == epochs and ep > 1:
            should_save = True
            save_reason = "last epoch"

        if should_save:
            _save_checkpoint(ep, score, save_reason)
            epoch_report_path = os.path.join(checkpoints_dir, f"epoch_{ep}", "dev_report.json")
            file_utils.save_json(epoch_report_path, dev_report)
        else:
            print(f"[INFO] Epoch {ep} not saved (current: {score:.4f}, best: {best_score:.4f} @ epoch {best_epoch})")

    saved_epochs = []
    if 1 in epoch_scores:
        saved_epochs.append({"epoch": 1, "reason": "first", "score": epoch_scores[1]})
    if best_epoch > 0 and best_epoch not in [1, epochs]:
        saved_epochs.append({"epoch": best_epoch, "reason": "best", "score": best_score})
    if epochs > 1:
        saved_epochs.append({"epoch": epochs, "reason": "last", "score": epoch_scores.get(epochs, float("nan"))})

    file_utils.save_json(os.path.join(output_dir, "training_summary.json"), {
        "total_epochs": epochs,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metric": "level1.overall.micro_f1",
        "all_epoch_scores": {f"epoch_{k}": v for k, v in epoch_scores.items()},
        "saved_checkpoints": saved_epochs,
        "output_structure": {
            "best/": "Best performing checkpoint across all epochs",
            "checkpoints/epoch_N/": "Saved only for: first epoch, new best, and last epoch",
        },
    })

    print(f"\n{'=' * 60}\nTraining Complete!\n{'=' * 60}")
    print(f"Best epoch: {best_epoch} (score: {best_score:.4f})")
    print(f"Best model: {best_dir}")
    print(f"Saved checkpoints: {[ckpt['epoch'] for ckpt in saved_epochs]}")
    print(f"Checkpoints dir: {checkpoints_dir}")
    print(f"{'=' * 60}\n")

    return model, tok


# ---------------------------------------------------------------------
# Ablation Study Helpers
# ---------------------------------------------------------------------

ABLATION_VARIANTS_DEFAULT = "wo_actor,wo_evidence,wo_context,unit_text_only"

_ABLATION_ALIASES = {
    "full": "full",
    "none": "full",
    "original": "full",
    "wo_actor": "wo_actor",
    "w_o_actor": "wo_actor",
    "without_actor": "wo_actor",
    "no_actor": "wo_actor",
    "wo_evidence": "wo_evidence",
    "w_o_evidence": "wo_evidence",
    "without_evidence": "wo_evidence",
    "no_evidence": "wo_evidence",
    "wo_context": "wo_context",
    "w_o_context": "wo_context",
    "without_context": "wo_context",
    "no_context": "wo_context",
    "unit_text_only": "unit_text_only",
    "text_only": "unit_text_only",
    "unit_only": "unit_text_only",
}


def _normalize_ablation_variant(x: str) -> str:
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = s.replace("w/o", "wo")
    s = s.replace("w\\o", "wo")
    if s in _ABLATION_ALIASES:
        return _ABLATION_ALIASES[s]
    valid = sorted(set(_ABLATION_ALIASES.values()))
    raise ValueError(f"Unknown ablation variant: {x}. Valid variants: {valid}")


def parse_ablation_variants(x: str, include_full: bool = False) -> List[str]:
    raw = [v.strip() for v in str(x).split(",") if v.strip()]
    variants = [_normalize_ablation_variant(v) for v in raw]
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            out.append(v)
            seen.add(v)
    if include_full and "full" not in seen:
        out = ["full"] + out
    return out


def _is_actor_key(k: str) -> bool:
    k = str(k).lower()
    return "actor" in k


def _is_evidence_key(k: str) -> bool:
    k = str(k).lower()
    return any(t in k for t in [
        "evidence",
        "quote",
        "supporting_sentence",
        "supporting_sentences",
        "grounded_sentence",
        "source_sentence",
        "source_sentences",
    ])


def _is_context_key(k: str) -> bool:
    k = str(k).lower()
    return any(t in k for t in [
        "context",
        "neighbor",
        "neighbour",
        "surrounding",
        "background",
        "article_title",
        "title",
        "article_content",
        "content",
        "full_article",
        "article_text",
    ])


def _empty_like(v, empty_text: str = ""):
    if isinstance(v, str):
        return empty_text
    if isinstance(v, (list, tuple, set)):
        return []
    if isinstance(v, dict):
        return {}
    return empty_text


def _mask_actor_like(v, placeholder: str = "[ACTOR_MASKED]"):
    if isinstance(v, str):
        return placeholder
    if isinstance(v, (list, tuple, set)):
        return []
    if isinstance(v, dict):
        return {}
    return placeholder


def _recursive_replace_by_key(obj, key_predicate, replace_fn):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if key_predicate(k):
                new_obj[k] = replace_fn(v)
            else:
                new_obj[k] = _recursive_replace_by_key(v, key_predicate, replace_fn)
        return new_obj
    if isinstance(obj, list):
        return [_recursive_replace_by_key(x, key_predicate, replace_fn) for x in obj]
    return obj


def apply_input_ablation_to_payload(
    payload: Dict[str, Any],
    variant: str,
    actor_placeholder: str = "[ACTOR_MASKED]",
    empty_text: str = "",
) -> Dict[str, Any]:
    variant = _normalize_ablation_variant(variant)
    p = copy.deepcopy(payload)

    if variant == "full":
        return p

    if variant == "wo_actor":
        return _recursive_replace_by_key(p, _is_actor_key, lambda v: _mask_actor_like(v, actor_placeholder))

    if variant == "wo_evidence":
        return _recursive_replace_by_key(p, _is_evidence_key, lambda v: _empty_like(v, empty_text))

    if variant == "wo_context":
        return _recursive_replace_by_key(p, _is_context_key, lambda v: _empty_like(v, empty_text))

    if variant == "unit_text_only":
        p = _recursive_replace_by_key(p, _is_actor_key, lambda v: _mask_actor_like(v, actor_placeholder))
        p = _recursive_replace_by_key(p, _is_evidence_key, lambda v: _empty_like(v, empty_text))
        p = _recursive_replace_by_key(p, _is_context_key, lambda v: _empty_like(v, empty_text))
        return p

    raise ValueError(f"Unsupported ablation variant: {variant}")


@contextmanager
def patch_build_input_payload_for_ablation(
    variant: str,
    actor_placeholder: str = "[ACTOR_MASKED]",
    empty_text: str = "",
):
    original_data_utils_fn = data_utils.build_input_payload

    def patched_build_input_payload(*args, **kwargs):
        payload = original_data_utils_fn(*args, **kwargs)
        return apply_input_ablation_to_payload(
            payload=payload,
            variant=variant,
            actor_placeholder=actor_placeholder,
            empty_text=empty_text,
        )

    patched_slots = []
    data_utils.build_input_payload = patched_build_input_payload
    patched_slots.append((data_utils, "build_input_payload", original_data_utils_fn))

    if hasattr(infer, "data_utils") and hasattr(infer.data_utils, "build_input_payload"):
        original_infer_du_fn = infer.data_utils.build_input_payload
        infer.data_utils.build_input_payload = patched_build_input_payload
        patched_slots.append((infer.data_utils, "build_input_payload", original_infer_du_fn))

    if hasattr(infer, "build_input_payload"):
        original_infer_direct_fn = infer.build_input_payload
        infer.build_input_payload = patched_build_input_payload
        patched_slots.append((infer, "build_input_payload", original_infer_direct_fn))

    try:
        yield
    finally:
        for obj, name, old_fn in reversed(patched_slots):
            setattr(obj, name, old_fn)


def save_pred_and_report(
    *,
    pred: Dict[InstanceId, Dict[str, Set[str]]],
    gold_l1: Dict[InstanceId, Dict[str, Set[str]]],
    pred_out: str,
    report_out: str,
    l1tol2_mapping_path: str,
    file_utils,
    eval_module,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(pred_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)

    serializable = []
    for iid, d in pred.items():
        guid, unit_level, unit_id, actor = iid
        serializable.append({
            "guid": guid,
            "unit_level": unit_level,
            "unit_id": unit_id,
            "actor": actor,
            "aligned_with_human_values": sorted(list(d.get("aligned", set()))),
            "contradictory_to_human_values": sorted(list(d.get("contradictory", set()))),
        })

    file_utils.save_json(pred_out, serializable)
    report = eval_module.build_eval_report_l1_and_l2_from_l1(
        gold_l1=gold_l1,
        pred_l1=pred,
        l1tol2_mapping_path=l1tol2_mapping_path,
    )
    file_utils.save_json(report_out, report)
    return report


def _compact_report_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for section in ["level1", "level2_from_level1"]:
        overall = get_nested(report, [section, "overall"], default={}) or {}
        out[section] = {
            "micro_f1": overall.get("micro_f1"),
            "macro_f1": overall.get("macro_f1"),
            "micro_precision": overall.get("micro_precision"),
            "micro_recall": overall.get("micro_recall"),
            "macro_precision": overall.get("macro_precision"),
            "macro_recall": overall.get("macro_recall"),
            "drr": overall.get("drr"),
            "support_instances": overall.get("support_instances"),
            "support_gold_labels": overall.get("support_gold_labels"),
            "support_pred_labels": overall.get("support_pred_labels"),
        }
    return out


def run_one_ablation_inference(
    *,
    args: Namespace,
    model,
    tok,
    test_events_by_guid: Dict[str, Dict[str, Any]],
    test_gold: Dict[InstanceId, Dict[str, Set[str]]],
    hv_label_names_str: str,
    hv_label_description_str: str,
    prompt_content: str,
    prompt_system: str,
) -> Dict[InstanceId, Dict[str, Set[str]]]:
    if args.group == "G1":
        if args.g1_evaluate_only == "y":
            raise ValueError("Ablation mode needs fresh prompts. Set --g1_evaluate_only n for G1 ablation.")

        return infer.run_inference(
            group="G1",
            provider=args.provider,
            model_name=args.model_name,
            api_key=args.api_key,
            events_by_guid=test_events_by_guid,
            gold=test_gold,
            label_space=args.label_space,
            hv_label=hv_label_names_str,
            hv_label_desc=hv_label_description_str,
            prompt_content=prompt_content,
            prompt_system=prompt_system,
            prompt_variant=args.prompt_variant,
            local_model=None,
            local_tokenizer=None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_len=args.max_len,
            infer_batch_size=args.infer_batch_size,
            echv_exp_path=args.echv_exp_path,
            g1_use_map_reduce=True,
            role1=args.role1,
            deepseek_base_url=args.provider,
        )

    if args.group in ("G2", "G3"):
        return infer.run_inference(
            group=args.group,
            provider="",
            model_name=args.model_name,
            api_key="",
            events_by_guid=test_events_by_guid,
            gold=test_gold,
            label_space=args.label_space,
            hv_label=hv_label_names_str,
            hv_label_desc=hv_label_description_str,
            prompt_content=prompt_content,
            prompt_system=prompt_system,
            prompt_variant=args.prompt_variant,
            local_model=model,
            local_tokenizer=tok,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_len=args.max_len,
            infer_batch_size=args.infer_batch_size,
        )

    raise ValueError(f"Ablation mode supports G1/G2/G3 only, got group={args.group}")


def run_ablation_study(
    *,
    args: Namespace,
    hf_token: Optional[str],
    use_auto_map: bool,
    test_events_by_guid: Dict[str, Dict[str, Any]],
    test_gold: Dict[InstanceId, Dict[str, Set[str]]],
    hv_label_names_str: str,
    hv_label_description_str: str,
    prompt_content: str,
    prompt_system: str,
    file_utils,
    eval_module,
) -> Dict[str, Any]:
    if args.group == "G4":
        raise ValueError("Ablation mode is for G1/G2/G3 prompt/input ablations. G4 retrieval does not use prompts.")

    variants = parse_ablation_variants(args.ablation_variants, include_full=str2bool(args.ablation_include_full))
    ablation_out_dir = args.ablation_out_dir.strip() or os.path.join(args.out_dir, "ablation")
    os.makedirs(ablation_out_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("🔬 Ablation Study")
    print(f"Model group : {args.group}")
    print(f"Model name  : {args.model_name}")
    print(f"Variants    : {variants}")
    print(f"Output dir  : {ablation_out_dir}")
    print(f"{'=' * 70}\n")

    model, tok = None, None
    if args.group in ("G2", "G3"):
        model, tok = load_model_for_infer(
            group=args.group,
            model_name=args.model_name,
            use_auto_map=use_auto_map,
            hf_token=hf_token,
            adapter_dir=args.adapter_dir,
            load_dir=args.load_dir,
            merge_lora_infer=str2bool(args.merge_lora_infer),
        )

    summary = {
        "mode": "ablation",
        "group": args.group,
        "model_name": args.model_name,
        "label_space": args.label_space,
        "variants": {},
        "ablation_out_dir": ablation_out_dir,
        "support_instances": len(test_gold),
    }

    for variant in variants:
        variant_dir = os.path.join(ablation_out_dir, variant)
        pred_out = os.path.join(variant_dir, "test_pred.json")
        report_out = os.path.join(variant_dir, "test_report.json")

        print(f"\n{'-' * 70}")
        print(f"Running ablation variant: {variant}")
        print(f"pred_out   : {pred_out}")
        print(f"report_out : {report_out}")
        print(f"{'-' * 70}\n")

        with patch_build_input_payload_for_ablation(
            variant=variant,
            actor_placeholder=args.ablation_actor_placeholder,
            empty_text=args.ablation_empty_text,
        ):
            pred = run_one_ablation_inference(
                args=args,
                model=model,
                tok=tok,
                test_events_by_guid=test_events_by_guid,
                test_gold=test_gold,
                hv_label_names_str=hv_label_names_str,
                hv_label_description_str=hv_label_description_str,
                prompt_content=prompt_content,
                prompt_system=prompt_system,
            )

        report = save_pred_and_report(
            pred=pred,
            gold_l1=test_gold,
            pred_out=pred_out,
            report_out=report_out,
            l1tol2_mapping_path=args.l1tol2_mappings,
            file_utils=file_utils,
            eval_module=eval_module,
        )

        summary["variants"][variant] = {
            "pred_out": pred_out,
            "report_out": report_out,
            "metrics": _compact_report_summary(report),
        }
        print(json.dumps(_compact_report_summary(report), ensure_ascii=False, indent=2))

    summary_path = os.path.join(ablation_out_dir, "ablation_summary.json")
    file_utils.save_json(summary_path, summary)

    print(f"\n{'=' * 70}")
    print(" Ablation Study Complete!")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 70}\n")
    return summary


# ---------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------

def run_g4_retrieval(args, train_events_by_guid, train_gold, test_events_by_guid, test_gold, hv_label_names_str, hv_label_description_str):
    if nonllm_infer is None:
        raise ImportError("nonllm_retrieval.py is required for G4, but it could not be imported.")

    return nonllm_infer.run_retrieval_baseline(
        method=args.retrieval_method,
        train_events_by_guid=train_events_by_guid,
        train_gold=train_gold,
        target_events_by_guid=test_events_by_guid,
        target_gold=test_gold,
        label_space=args.label_space,
        hv_label=hv_label_names_str,
        hv_label_desc=hv_label_description_str,
        data_utils=data_utils,
        level_filter=args.retrieval_level_filter,
        k=args.retrieval_k,
        vote_threshold=args.retrieval_vote_threshold,
        min_sim=args.retrieval_min_sim,
        top_m_aligned=args.retrieval_top_m_aligned,
        top_m_contra=args.retrieval_top_m_contra,
        conflict_mode=args.retrieval_conflict_mode,
        max_chars_per_field=args.retrieval_max_chars_per_field,
        include_event_content_for_article=args.retrieval_include_article_content,
        tfidf_ngram_max=args.tfidf_ngram_max,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_max_df=args.tfidf_max_df,
        tfidf_max_features=args.tfidf_max_features,
        sbert_model_name=args.sbert_model_name,
        sbert_batch_size=args.sbert_batch_size,
        sbert_device=args.sbert_device,
        sbert_cache_folder=args.sbert_cache_folder,
    )


def run_regular_inference(args, hf_token, use_auto_map, train_events_by_guid, train_gold, test_events_by_guid, test_gold, hv_label_names_str, hv_label_description_str, prompt_content, prompt_system):
    if args.group == "G1":
        if args.g1_evaluate_only == "n":
            pred = infer.run_inference(
                group="G1",
                provider=args.provider,
                model_name=args.model_name,
                api_key=args.api_key,
                events_by_guid=test_events_by_guid,
                gold=test_gold,
                label_space=args.label_space,
                hv_label=hv_label_names_str,
                hv_label_desc=hv_label_description_str,
                prompt_content=prompt_content,
                prompt_system=prompt_system,
                prompt_variant=args.prompt_variant,
                local_model=None,
                local_tokenizer=None,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                max_len=args.max_len,
                infer_batch_size=args.infer_batch_size,
                echv_exp_path=args.echv_exp_path,
                g1_use_map_reduce=True,
                role1=args.role1,
                deepseek_base_url=args.provider,
            )
        else:
            folder_names = list(range(50))
            all_records: List[Dict[str, Any]] = []
            for folder_name in folder_names:
                tmp_result_path = args.echv_exp_path + args.group + "/output/" + args.model_name + "/" + str(folder_name) + "/output_results.json"
                data = data_utils.load_json(tmp_result_path)
                if not isinstance(data, list):
                    print(f"[WARN] {tmp_result_path} is not a list, skip.")
                    continue
                all_records.extend(data)
            print(f" Total records: {len(all_records)}")
            pred = data_utils.build_pred_from_records(all_records)
            print(f" Unique InstanceId (pred size): {len(pred)}")
            empty_cnt = data_utils.count_empty_pred_instances(pred)
            print(f"[PRED] empty (aligned+contradictory both empty): {empty_cnt} / {len(pred)}")

    elif args.group in ("G2", "G3"):
        model, tok = load_model_for_infer(
            group=args.group,
            model_name=args.model_name,
            use_auto_map=use_auto_map,
            hf_token=hf_token,
            adapter_dir=args.adapter_dir,
            load_dir=args.load_dir,
            merge_lora_infer=str2bool(args.merge_lora_infer),
        )
        pred = infer.run_inference(
            group=args.group,
            provider="",
            model_name=args.model_name,
            api_key="",
            events_by_guid=test_events_by_guid,
            gold=test_gold,
            label_space=args.label_space,
            hv_label=hv_label_names_str,
            hv_label_desc=hv_label_description_str,
            prompt_content=prompt_content,
            prompt_system=prompt_system,
            prompt_variant=args.prompt_variant,
            local_model=model,
            local_tokenizer=tok,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_len=args.max_len,
            infer_batch_size=args.infer_batch_size,
        )

    elif args.group == "G4":
        pred = run_g4_retrieval(
            args,
            train_events_by_guid,
            train_gold,
            test_events_by_guid,
            test_gold,
            hv_label_names_str,
            hv_label_description_str,
        )
    else:
        raise ValueError(f"Unknown group: {args.group}")

    report = save_pred_and_report(
        pred=pred,
        gold_l1=test_gold,
        pred_out=args.pred_out,
        report_out=args.report_out,
        l1tol2_mapping_path=args.l1tol2_mappings,
        file_utils=file_utils,
        eval_module=eval,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n{'=' * 70}")
    print(" Inference Complete!")
    print(f" Test predictions: {args.pred_out}")
    print(f" Test report: {args.report_out}")
    print(f"{'=' * 70}\n")
    return pred, report


def run_train_lora(args, hf_token, use_auto_map, train_events_by_guid, train_gold, dev_events_by_guid, dev_gold, test_events_by_guid, test_gold, hv_label_names_str, hv_label_description_str, prompt_content, prompt_system):
    assert args.group == "G3", "train_lora is only for G3"
    assert torch is not None, "torch not installed"
    assert AutoTokenizer is not None and AutoModelForCausalLM is not None, "transformers not installed"

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_examples = build_sft_examples(
        events_by_guid=train_events_by_guid,
        gold=train_gold,
        label_space=args.label_space,
        hv_label=hv_label_names_str,
        hv_label_desc=hv_label_description_str,
        prompt_content=prompt_content,
        prompt_system=prompt_system,
        prompt_variant=args.prompt_variant,
        model_name=args.model_name,
        group=args.group,
    )
    print("[INFO] SFT examples:", len(train_examples))
    train_ds = SFTDataset(train_examples, tok, max_len=args.max_len)

    def debug_check_mask(train_ds, tok, n=3):
        for _ in range(min(n, len(train_ds))):
            item = train_ds[random.randrange(len(train_ds))]
            labels = item["labels"].tolist()
            input_ids = item["input_ids"].tolist()
            first_sup = next((i for i, x in enumerate(labels) if x != -100), None)
            print("first supervised pos:", first_sup, "total_len:", len(labels))
            if first_sup is not None:
                print("prompt_tail:", tok.decode(input_ids[max(0, first_sup - 80):first_sup], skip_special_tokens=True))
                print("target_head:", tok.decode(input_ids[first_sup:first_sup + 120], skip_special_tokens=True))
            print("-" * 60)

    debug_check_mask(train_ds, tok, n=3)

    model, tok = train_lora_sft(
        args=args,
        base_model_name=args.model_name,
        output_dir=args.out_dir,
        train_dataset=train_ds,
        dev_gold=dev_gold,
        dev_events_by_guid=dev_events_by_guid,
        label_space=args.label_space,
        hv_label=hv_label_names_str,
        hv_label_desc=hv_label_description_str,
        prompt_content=prompt_content,
        prompt_system=prompt_system,
        prompt_variant=args.prompt_variant,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_new_tokens_eval=args.max_new_tokens,
    )
    maybe_save_merged_full_model(model, tok, args.save_merged_dir)

    best_adapter_dir = os.path.join(args.out_dir, "best")
    use_adapter_dir = best_adapter_dir if os.path.isdir(best_adapter_dir) else args.out_dir
    model_best, tok_best = load_model_for_infer(
        group="G3",
        model_name=args.model_name,
        use_auto_map=use_auto_map,
        hf_token=hf_token,
        adapter_dir=use_adapter_dir,
        load_dir="",
        merge_lora_infer=False,
    )

    test_pred = infer.run_inference(
        group="G3",
        provider="",
        model_name=args.model_name + "+LoRA(best)",
        api_key="",
        events_by_guid=test_events_by_guid,
        gold=test_gold,
        label_space=args.label_space,
        hv_label=hv_label_names_str,
        hv_label_desc=hv_label_description_str,
        prompt_content=prompt_content,
        prompt_system=prompt_system,
        prompt_variant=args.prompt_variant,
        local_model=model_best,
        local_tokenizer=tok_best,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_len=args.max_len,
        infer_batch_size=args.infer_batch_size,
    )

    report = save_pred_and_report(
        pred=test_pred,
        gold_l1=test_gold,
        pred_out=args.pred_out,
        report_out=args.report_out,
        l1tol2_mapping_path=args.l1tol2_mappings,
        file_utils=file_utils,
        eval_module=eval,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n{'=' * 70}")
    print("Training & Evaluation Complete!")
    print(f"All outputs saved to: {args.out_dir}")
    print(f"Test predictions: {args.pred_out}")
    print(f"Test report: {args.report_out}")
    print(f"Best model: {os.path.join(args.out_dir, 'best')}")
    print(f"{'=' * 70}\n")
    return test_pred, report


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/unified_value_recognition.json")

    # data
    ap.add_argument("--train_event", type=str, required=True)
    ap.add_argument("--train_hv", type=str, required=True)
    ap.add_argument("--dev_event", type=str, required=True)
    ap.add_argument("--dev_hv", type=str, required=True)
    ap.add_argument("--test_event", type=str, required=True)
    ap.add_argument("--test_hv", type=str, required=True)

    # run mode
    ap.add_argument("--mode", type=str, choices=["infer", "train_lora", "ablation"], required=True)

    # group/model
    ap.add_argument("--group", type=str, choices=["G1", "G2", "G3", "G4"], required=True)
    ap.add_argument("--provider", type=str, default="https://api.deepseek.com/beta", help="for G1: openai/anthropic/gemini/deepseek")
    ap.add_argument("--model_name", type=str, default="", help="Model name for G1/G2/G3. For G4, optional; retrieval_method is used if empty.")
    ap.add_argument("--api_key", type=str, default="")

    # task
    ap.add_argument("--label_space", type=str, choices=["L1", "L2"], default="L1")
    ap.add_argument("--prompt_variant", type=str, default="A")
    ap.add_argument("--max_new_tokens", type=int, default=256)

    # lora training
    ap.add_argument("--out_dir", type=str, default="./outputs/lora_ckpt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)

    # outputs
    ap.add_argument("--pred_out_dev", type=str, default="./outputs/pred_dev.json")
    ap.add_argument("--report_out_dev", type=str, default="./outputs/report_dev.json")
    ap.add_argument("--pred_out", type=str, default="./outputs/pred_test.json")
    ap.add_argument("--report_out", type=str, default="./outputs/report_test.json")

    # GPU
    ap.add_argument("--gpu_ids", type=str, default="", help="e.g. '0' or '0,1'. If set, overrides CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--device_map", type=str, default="single", choices=["single", "auto"], help="single=force single GPU; auto=transformers device_map='auto'")

    # sampling/debug
    ap.add_argument("--sample_train_n", type=int, default=0)
    ap.add_argument("--sample_dev_n", type=int, default=0)
    ap.add_argument("--sample_test_n", type=int, default=0)
    ap.add_argument("--sample_seed", type=int, default=42)

    # paths/config
    ap.add_argument("--adapter_dir", type=str, default="", help="For G3 infer: path to LoRA adapter dir.")
    ap.add_argument("--load_dir", type=str, default="", help="For infer: load full finetuned checkpoint from this dir.")
    ap.add_argument("--merge_lora_infer", type=str, default="0", help="For G3 infer with adapter_dir: merge LoRA into base before inference. 1/0")
    ap.add_argument("--save_merged_dir", type=str, default="", help="For train_lora: additionally save merged full model to this dir.")
    ap.add_argument("--hf_token", type=str, default="", help="Optional HuggingFace token, or set HUGGINGFACE_HUB_TOKEN.")
    ap.add_argument("--hv1_label_dir", type=str, default="")
    ap.add_argument("--hv_label_dir", type=str, default="")
    ap.add_argument("--l1tol2_mappings", type=str, default="")
    ap.add_argument("--canonical_prompt", type=str, default="")
    ap.add_argument("--canonical_system", type=str, default="")
    ap.add_argument("--infer_batch_size", type=int, default=4)
    ap.add_argument("--dump_sampled_dir", type=str, default="dataset/")
    ap.add_argument("--use_timestamp", type=str, default="1")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--g1_evaluate_only", type=str, default="n")
    ap.add_argument("--echv_exp_path", type=str, default="n")
    ap.add_argument("--g1_use_map_reduce", type=str2bool, default=False)
    ap.add_argument("--role1", type=str, default="developer")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--resume_lora_dir", type=str, default="")
    ap.add_argument("--resume_epoch", type=int, default=0)

    # G4 retrieval
    ap.add_argument("--retrieval_method", type=str, default="tfidf", choices=["tfidf", "sbert"])
    ap.add_argument("--retrieval_k", type=int, default=5)
    ap.add_argument("--retrieval_vote_threshold", type=float, default=0.3)
    ap.add_argument("--retrieval_min_sim", type=float, default=0.0)
    ap.add_argument("--retrieval_level_filter", type=str2bool, default=True)
    ap.add_argument("--retrieval_conflict_mode", type=str, default="prefer_higher", choices=["prefer_higher", "drop", "prefer_aligned", "prefer_contradictory"])
    ap.add_argument("--retrieval_top_m_aligned", type=int, default=0)
    ap.add_argument("--retrieval_top_m_contra", type=int, default=0)
    ap.add_argument("--retrieval_max_chars_per_field", type=int, default=6000)
    ap.add_argument("--retrieval_include_article_content", type=str2bool, default=False)
    ap.add_argument("--tfidf_ngram_max", type=int, default=2)
    ap.add_argument("--tfidf_min_df", type=int, default=2)
    ap.add_argument("--tfidf_max_df", type=float, default=0.95)
    ap.add_argument("--tfidf_max_features", type=int, default=200000)
    ap.add_argument("--sbert_model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--sbert_batch_size", type=int, default=32)
    ap.add_argument("--sbert_device", type=str, default="")
    ap.add_argument("--sbert_cache_folder", type=str, default="")

    # Ablation study
    ap.add_argument("--ablation_variants", type=str, default=ABLATION_VARIANTS_DEFAULT,
                    help="Comma-separated ablation variants: wo_actor,wo_evidence,wo_context,unit_text_only. Optional: full.")
    ap.add_argument("--ablation_include_full", type=str, default="0",
                    help="If 1, also run the original full-input setting before ablations.")
    ap.add_argument("--ablation_out_dir", type=str, default="",
                    help="Output root for ablation results. Empty = args.out_dir/ablation.")
    ap.add_argument("--ablation_actor_placeholder", type=str, default="[ACTOR_MASKED]")
    ap.add_argument("--ablation_empty_text", type=str, default="")

    # -----------------------------------------------------------------
    # Keep your preferred DEBUG=True style.
    # Change only these sys.argv values when debugging.
    # -----------------------------------------------------------------
    DEBUG = True
    if DEBUG:
        sys.argv = [sys.argv[0]] + [
            "--mode", "ablation",
            "--group", "G2",
            "--gpu_ids", "0",
            "--device_map", "auto",

            "--model_name", "meta-llama/Llama-3.1-8B-Instruct",
            "--adapter_dir", "",

            "--label_space", "L1",
            "--prompt_variant", "A",
            "--epochs", "1",
            "--lr", "2e-4",
            "--batch_size", "1",
            "--grad_accum", "8",
            "--max_len", "4096",
            "--max_new_tokens", "128",
            "--infer_batch_size", "4",

            "--train_event", "dataset/training_dataset_total_formatted.json",
            "--train_hv", "dataset/train_sub_gold.json",
            "--dev_event", "dataset/dev_dataset_total_formatted.json",
            "--dev_hv", "dataset/dev_sub_gold.json",
            "--test_event", "dataset/test_dataset_total_formatted.json",
            "--test_hv", "dataset/test_sub_gold.json",

            "--canonical_prompt", "prompt/canonical_prompt_openai2.txt",
            "--canonical_system", "prompt/canonical_system.txt",
            "--hv1_label_dir", "dataset/hv/id2hv.json",
            "--hv_label_dir", "dataset/hv/values.json",
            "--l1tol2_mappings", "dataset/hv/l1tol2_id_mapping.json",

            "--sample_train_n", "999999",
            "--sample_dev_n", "999999",
            "--sample_test_n", "999999",
            "--sample_seed", "42",
            "--use_timestamp", "0",

            "--out_dir", "results/ablation/llama31_lora",
            "--ablation_out_dir", "results/ablation/llama31_lora",
            "--ablation_variants", "wo_actor,wo_evidence,wo_context,unit_text_only",
            "--ablation_include_full", "1",
        ]

    args = ap.parse_args()

    if args.group in ("G1", "G2", "G3") and not args.model_name:
        ap.error("--model_name is required for G1/G2/G3.")
    if args.group == "G4" and not args.model_name:
        args.model_name = args.retrieval_method
    if args.mode == "ablation" and args.group == "G4":
        ap.error("--mode ablation supports G1/G2/G3 only, because G4 does not use prompt payloads.")

    hf_token = args.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN", None)

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    use_auto_map = (args.device_map == "auto")

    if args.resume_lora_dir:
        resume_run_dir = _resolve_run_dir_from_resume_path(args.resume_lora_dir)
        args.out_dir = resume_run_dir
        args.pred_out = os.path.join(args.out_dir, "test_pred.json")
        args.report_out = os.path.join(args.out_dir, "test_report.json")
        args.pred_out_dev = os.path.join(args.out_dir, "dev_pred.json")
        args.report_out_dev = os.path.join(args.out_dir, "dev_report.json")
        args.use_timestamp = "0"
        print(f"\n{'=' * 70}")
        print("Resume detected. Reusing run directory:")
        print(f"  resume_from: {args.resume_lora_dir}")
        print(f"  run_dir    : {args.out_dir}")
        print(f"{'=' * 70}\n")

    if str2bool(args.use_timestamp):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if args.run_name:
            dir_suffix = f"{args.run_name}_{timestamp}"
        else:
            model_short = args.model_name.split("/")[-1][:20]
            dir_suffix = f"{model_short}_{timestamp}"
        base_out_dir = os.path.dirname(args.out_dir) or "outputs"
        args.out_dir = os.path.join(base_out_dir, dir_suffix)
        args.pred_out = os.path.join(args.out_dir, "test_pred.json")
        args.report_out = os.path.join(args.out_dir, "test_report.json")
        args.pred_out_dev = os.path.join(args.out_dir, "dev_pred.json")
        args.report_out_dev = os.path.join(args.out_dir, "dev_report.json")
        print(f"\n{'=' * 70}")
        print(f"Run ID: {dir_suffix}")
        print(f"Output directory: {args.out_dir}")
        print(f"{'=' * 70}\n")
    else:
        print(f"\n{'=' * 70}")
        print(f"Using fixed output directory: {args.out_dir}")
        print("Warning: This will overwrite previous results!")
        print(f"{'=' * 70}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    run_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "group": args.group,
        "model_name": args.model_name,
        "label_space": args.label_space,
        "epochs": args.epochs if args.mode == "train_lora" else None,
        "lr": args.lr if args.mode == "train_lora" else None,
        "batch_size": args.batch_size if args.mode == "train_lora" else None,
        "max_len": args.max_len,
        "max_new_tokens": args.max_new_tokens,
        "train_data": args.train_hv,
        "dev_data": args.dev_hv,
        "test_data": args.test_hv,
        "gpu_ids": args.gpu_ids,
        "command": " ".join(sys.argv),
        "retrieval_method": args.retrieval_method if args.group == "G4" else None,
        "retrieval_k": args.retrieval_k if args.group == "G4" else None,
        "retrieval_vote_threshold": args.retrieval_vote_threshold if args.group == "G4" else None,
        "retrieval_level_filter": args.retrieval_level_filter if args.group == "G4" else None,
        "sbert_model_name": args.sbert_model_name if args.group == "G4" and args.retrieval_method == "sbert" else None,
        "ablation_variants": args.ablation_variants if args.mode == "ablation" else None,
        "ablation_include_full": args.ablation_include_full if args.mode == "ablation" else None,
        "ablation_out_dir": args.ablation_out_dir if args.mode == "ablation" else None,
    }

    # load data
    train_event = file_utils.load_json(args.train_event)
    dev_event = file_utils.load_json(args.dev_event)
    test_event = file_utils.load_json(args.test_event)
    train_gold_raw = file_utils.load_json(args.train_hv)
    dev_gold_raw = file_utils.load_json(args.dev_hv)
    test_gold_raw = file_utils.load_json(args.test_hv)

    train_events_by_guid = data_utils.index_events_by_guid(train_event)
    dev_events_by_guid = data_utils.index_events_by_guid(dev_event)
    test_events_by_guid = data_utils.index_events_by_guid(test_event)

    train_gold = data_utils.list_to_gold_dict(train_gold_raw)
    dev_gold = data_utils.list_to_gold_dict(dev_gold_raw)
    test_gold = data_utils.list_to_gold_dict(test_gold_raw)

    if args.sample_train_n > 0:
        train_gold = data_utils.sample_gold_instances(train_gold, args.sample_train_n, args.sample_seed)
        train_events_by_guid = data_utils.filter_events_by_gold(train_events_by_guid, train_gold)
    if args.sample_dev_n > 0:
        dev_gold = data_utils.sample_gold_instances(dev_gold, args.sample_dev_n, args.sample_seed + 1)
        dev_events_by_guid = data_utils.filter_events_by_gold(dev_events_by_guid, dev_gold)
    if args.sample_test_n > 0:
        test_gold = data_utils.sample_gold_instances(test_gold, args.sample_test_n, args.sample_seed + 2)
        test_events_by_guid = data_utils.filter_events_by_gold(test_events_by_guid, test_gold)

    if args.dump_sampled_dir:
        try:
            dump_sampled_split_files(
                split_name="train",
                events_list=train_event,
                hv_rows_list=train_gold_raw,
                sampled_gold=train_gold,
                out_dir=args.dump_sampled_dir,
                file_utils=file_utils,
            )
            dump_sampled_split_files(
                split_name="dev",
                events_list=dev_event,
                hv_rows_list=dev_gold_raw,
                sampled_gold=dev_gold,
                out_dir=args.dump_sampled_dir,
                file_utils=file_utils,
            )
            dump_sampled_split_files(
                split_name="test",
                events_list=test_event,
                hv_rows_list=test_gold_raw,
                sampled_gold=test_gold,
                out_dir=args.dump_sampled_dir,
                file_utils=file_utils,
            )
        except Exception as e:
            print(f"[WARN] Failed to dump sampled split files: {e}")

    hv_label_names_str = prompt_utils.get_hv1_mappings(args.hv1_label_dir) if args.hv1_label_dir else ""
    hv_label_description_str = prompt_utils.get_hv_description_dict(args.hv_label_dir) if args.hv_label_dir else ""

    if args.group == "G4":
        prompt_content = ""
        prompt_system = ""
    else:
        prompt_content = file_utils.python_file_to_json(args.canonical_prompt)
        prompt_system = file_utils.python_file_to_json(args.canonical_system)

    print("[INFO] instances:", len(train_gold), len(dev_gold), len(test_gold))

    if args.mode == "ablation":
        ablation_summary = run_ablation_study(
            args=args,
            hf_token=hf_token,
            use_auto_map=use_auto_map,
            test_events_by_guid=test_events_by_guid,
            test_gold=test_gold,
            hv_label_names_str=hv_label_names_str,
            hv_label_description_str=hv_label_description_str,
            prompt_content=prompt_content,
            prompt_system=prompt_system,
            file_utils=file_utils,
            eval_module=eval,
        )
        run_config["ablation_summary"] = os.path.join(
            args.ablation_out_dir if args.ablation_out_dir else os.path.join(args.out_dir, "ablation"),
            "ablation_summary.json",
        )
        file_utils.save_json(os.path.join(args.out_dir, "run_config.json"), run_config)
        return

    if args.mode == "infer":
        run_regular_inference(
            args,
            hf_token,
            use_auto_map,
            train_events_by_guid,
            train_gold,
            test_events_by_guid,
            test_gold,
            hv_label_names_str,
            hv_label_description_str,
            prompt_content,
            prompt_system,
        )
        file_utils.save_json(os.path.join(args.out_dir, "run_config.json"), run_config)
        return

    if args.mode == "train_lora":
        run_train_lora(
            args,
            hf_token,
            use_auto_map,
            train_events_by_guid,
            train_gold,
            dev_events_by_guid,
            dev_gold,
            test_events_by_guid,
            test_gold,
            hv_label_names_str,
            hv_label_description_str,
            prompt_content,
            prompt_system,
        )
        file_utils.save_json(os.path.join(args.out_dir, "run_config.json"), run_config)
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
