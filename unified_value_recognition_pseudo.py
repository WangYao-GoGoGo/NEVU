#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified runner for:
- G1: API LLM inference (OpenAI / Claude / Gemini / DeepSeek)  -> no training
- G2: Open-source instruct model inference (Llama/Qwen/Ministral) -> no training
- G3: LoRA fine-tuning (Llama/Qwen) + inference + evaluation

Data:
- event_formatted.json: list[dict], keyed by guid, includes:
    - phase_version, filename
    - article fields (title/context etc.)
    - subevents list with id/text
    - behavior chains / story narratives (best-effort extraction)
    - actors: list[{actor_name: "guid-i"}]
- hv_rows_formatted.json: list[dict], each row is one hv label instance:
    {
      guid, filename, phase_version,
      unit_level: "article|subevent|behavior_chain|story_narrative",
      unit_id: "" or subevent_id or behavior_chain_id or story_narrative_id,
      actor: "guid-i" (already mapped),
      l1_human_value: int(0..53),
      l2_human_value: int(0..19),
      direction: 1(aligned) / 0(contradictory),
      ...
    }

Training:
- We train a single unified model across 4 levels by putting unit_level in the prompt.
- We evaluate overall and per level.

Metrics:
- micro-F1 and macro-F1 (gold-supported macro) similar to your evaluator.  (see evaluator inspiration) :contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set, Optional
from collections import defaultdict
from argparse import Namespace

import numpy as np
from tqdm import tqdm
from config import config_unified_value_recognition as config
import sys
from utils import prompt_utils
import inference_pseudo as infer
import evaluate as eval
from utils import file_utils
from utils import data_utils


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



import random
from typing import Iterable

from peft import PeftModel

# ===== ADD 1) 在 imports 里补充（文件顶部 import 区域） =====
from typing import Optional

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "t")

import random
from collections import defaultdict

def _count_unique_guids(events_list):
    return len({str(e.get("guid", "")).strip() for e in events_list if isinstance(e, dict) and str(e.get("guid","")).strip()})

import random
from collections import defaultdict
from typing import Any, Dict, Set, Tuple, List

import os, json, random
import numpy as np
import torch

def _resolve_run_dir_from_resume_path(resume_path: str) -> str:
    """
    resume_path can be:
      - .../<run>/checkpoints/epoch_4/
      - .../<run>/best/
      - .../<run>/checkpoints/
      - .../<run>/
    Return <run> (the run root dir).
    """
    p = os.path.abspath(resume_path).rstrip("/")

    base = os.path.basename(p)
    parent = os.path.basename(os.path.dirname(p))

    # case: .../checkpoints/epoch_N
    if re.match(r"^epoch_\d+$", base) and parent == "checkpoints":
        return os.path.dirname(os.path.dirname(p))  # up 2 -> <run>

    # case: .../best
    if base == "best":
        return os.path.dirname(p)  # up 1 -> <run>

    # case: .../checkpoints
    if base == "checkpoints":
        return os.path.dirname(p)  # up 1 -> <run>

    # default: assume user already passed <run>
    return p

def _rng_state_dict():
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st

def _set_rng_state_dict(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])

def save_checkpoint_lora(
    ckpt_dir: str,
    model, tokenizer,
    optimizer, scheduler,
    scaler=None,
    epoch: int = 0,
    global_step: int = 0,
    extra: dict | None = None,
):
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1) LoRA adapter (PEFT)
    model.save_pretrained(ckpt_dir)

    # 2) tokenizer (optional but recommended)
    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    # 3) training states
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
    model, optimizer=None, scheduler=None, scaler=None,
    map_location: str = "cpu",
):
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

    return state  # contains epoch/global_step/extra

def dump_sampled_split_files(
    *,
    split_name: str,
    events_list: List[Dict[str, Any]],
    hv_rows_list: List[Dict[str, Any]],
    sampled_gold: Dict[Tuple[str, str, str, str], Dict[str, Set[str]]],
    out_dir: str,
    file_utils,
) -> Tuple[str, str]:
    """
    根据 sampled_gold 的 iid 集合过滤 events_list 与 hv_rows_list，
    并输出两份文件：{split_name}_event_base.json / {split_name}_labels.json
    """
    os.makedirs(out_dir, exist_ok=True)

    keep_iids = set(sampled_gold.keys())
    keep_guids = {iid[0] for iid in keep_iids}

    # 1) events：按 guid 过滤
    sampled_events = []
    if isinstance(events_list, list):
        for e in events_list:
            if not isinstance(e, dict):
                continue
            g = str(e.get("guid", "")).strip()
            if g in keep_guids:
                sampled_events.append(e)

    # 2) hv_rows：按 (guid, unit_level, unit_id, actor) 精确过滤
    sampled_hv = []
    if isinstance(hv_rows_list, list):
        for r in hv_rows_list:
            if not isinstance(r, dict):
                continue
            iid = _norm_iid_from_hv_row(r)
            if iid in keep_iids:
                sampled_hv.append(r)

    # 输出文件
    event_path = os.path.join(out_dir, f"{split_name}_event_base.json")
    hv_path = os.path.join(out_dir, f"{split_name}_labels.json")

    file_utils.save_json(event_path, sampled_events)
    file_utils.save_json(hv_path, sampled_hv)

    print(f"[INFO] dumped {split_name}: events={len(sampled_events)}, hv_rows={len(sampled_hv)}")
    print(f"[INFO] -> {event_path}")
    print(f"[INFO] -> {hv_path}")
    return event_path, hv_path

# ===== ADD 2) 新增一个“统一加载模型”的函数（放在 main() 前即可） =====
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
        if not p or (not os.path.isdir(p)):
            return False
        cand = [
            "tokenizer.json", "tokenizer.model",
            "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt",
        ]
        return any(os.path.exists(os.path.join(p, x)) for x in cand)

    def _load_tokenizer(src: str):
        # 先走常规 HF tokenizer（Llama/Qwen/Phi 等走这条）
        try:
            return AutoTokenizer.from_pretrained(src, token=hf_token, use_fast=True)
        except Exception as e1:
            msg = str(e1)

            #  关键：Ministral/部分 Mistral3 系列会遇到 TokenizersBackend
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

            # 其他错误：再尝试 slow / trust_remote_code
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

        # 1) 优先按 CausalLM 加载（Llama/Qwen/Phi 等都走这里）
        try:
            return AutoModelForCausalLM.from_pretrained(src, **common_kwargs)
        except Exception as e_causal:
            last_err = e_causal

        # 2) Ministral-3 / Mistral3：Transformers 里常见是 Mistral3ForConditionalGeneration
        try:
            from transformers import Mistral3ForConditionalGeneration
            return Mistral3ForConditionalGeneration.from_pretrained(src, **common_kwargs)
        except Exception as e_m3:
            last_err = e_m3

        # 3) 最后兜底：trust_remote_code（少数 repo 需要）
        try:
            return AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True, **common_kwargs)
        except Exception as e_trc:
            last_err = e_trc

        # 4) 再兜底一次 Mistral3 + trust_remote_code（极少数情况）
        try:
            from transformers import Mistral3ForConditionalGeneration
            return Mistral3ForConditionalGeneration.from_pretrained(
                src, trust_remote_code=True, **common_kwargs
            )
        except Exception as e_trc2:
            last_err = e_trc2

        raise RuntimeError(f"Failed to load model from {src}. last_err={last_err}")

    # -----------------
    # tokenizer source：避免 G2 被 adapter_dir 污染
    # -----------------
    tok_candidates = []
    if load_dir:
        tok_candidates.append(load_dir)
    # 只有 G3 且 adapter_dir 真的含 tokenizer 文件时才允许用它
    if g == "G3" and adapter_dir and _has_tokenizer_files(adapter_dir):
        tok_candidates.append(adapter_dir)
    # 兜底永远是 base model
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

    # -----------------
    # model loading (原逻辑不变)
    # -----------------
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

# ===== ADD 3) 训练结束后可选保存 merged full model（放在 train_lora_sft 末尾 return 前） =====
def maybe_save_merged_full_model(model, tok, save_merged_dir: str):
    """
    Save merged full model (base+LoRA) for easier inference later.
    """
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

    if args.adapter_dir:  #  兼容 G3-LoRA
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        # 可选：合并后推理更快（合并后不再是LoRA形态）
        # model = model.merge_and_unload()
    else:                 #  兼容 G2 / ECHV-LLAMA(full)
        model = base

    model.eval()
    return model, tok


# ----------------------------
# Build instance-level gold labels from hv rows
#   instance_id = (guid, unit_level, unit_id, actor)
#   labels stored as sets for aligned / contradictory
# ----------------------------
InstanceId = Tuple[str, str, str, str]  # guid, unit_level, unit_id, actor
# ----------------------------
# Dataset for SFT (LoRA) training (G3)
# We train by next-token loss: prompt + gold_json
# ----------------------------
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

        # 1) 分开编码：保证 target 永远完整
        prompt_ids = self.tok(ex.prompt, add_special_tokens=False)["input_ids"]
        target_text = "\n" + ex.target_json
        target_ids = self.tok(target_text, add_special_tokens=False)["input_ids"]

        # 2) 如果超过 max_len：只截断 prompt（从左边截断，优先保留“输入末尾/近邻上下文”）
        if len(prompt_ids) + len(target_ids) > self.max_len:
            keep_prompt = max(0, self.max_len - len(target_ids))
            prompt_ids = prompt_ids[-keep_prompt:]  # 保留 prompt 的末尾
            # target_ids 保持完整

        if len(target_ids) > self.max_len:
            target_ids = target_ids[:self.max_len]  # 或 raise ValueError
            prompt_ids = []

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)

        # 3) labels：prompt 部分 -100，target 部分监督
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

# ----------------------------
# Build training examples from gold (multi-label per instance)
# ----------------------------
def gold_to_target_json(g, label_dropout: float = 0.0) -> str:
    a = list(g["aligned"])
    c = list(g["contradictory"])

    if label_dropout > 0:
        import random
        def drop(xs):
            if len(xs) <= 1:
                return xs
            kept = [x for x in xs if random.random() > label_dropout]
            return kept if kept else xs[:1]  # 至少保留1个，避免学到全空
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
        if "0-4562-2-1png" in guid:
            print("test")
        e = events_by_guid.get(guid)
        if not e:
            continue
        payload = data_utils.build_input_payload(e, iid, label_space, hv_label, hv_label_desc)
        prompt = prompt_utils.build_prompt_text(payload, prompt_variant, prompt_content, prompt_system, model_name, group)
        target = gold_to_target_json(g, label_dropout=label_dropout)
        examples.append(SFTExample(iid=iid, prompt=prompt, target_json=target))
    return examples

def get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def load_local_model_and_tokenizer(base_model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "Ministral" in base_model_name or "ministral" in base_model_name:
        #  官方推荐：mistral-common tokenizer backend + transformers v5
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

        tok = MistralCommonBackend.from_pretrained(base_model_name)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    else:
        # 其他模型沿用你的原逻辑
        tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    # 统一 tokenizer padding 处理
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    tok.padding_side = "left"

    return model, tok

# ----------------------------
# LoRA training loop (G3)
# ----------------------------
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
    # tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    tok.truncation_side = "left"
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    #  创建主输出目录
    os.makedirs(output_dir, exist_ok=True)

    #  Best checkpoint 目录
    best_dir = os.path.join(output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    #  Checkpoints 目录（只保存关键 epoch）
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

    #  记录所有 epoch 的分数（用于最后生成摘要）
    epoch_scores = {}

    def _pick_score(overall: Dict[str, Any]) -> float:
        """从 overall 里取出评估分数"""
        if not isinstance(overall, dict):
            return float("nan")

        candidates = [
            "micro_f1", "micro-F1", "micro_f1_overall", "micro_f1_gold_supported",
            "macro_f1", "macro-F1", "macro_f1_overall", "macro_f1_gold_supported",
        ]
        for k in candidates:
            v = overall.get(k, None)
            if isinstance(v, (int, float)):
                return float(v)

        # fallback：找第一个数值字段
        for k, v in overall.items():
            if isinstance(v, (int, float)):
                return float(v)

        return float("nan")

    def _save_checkpoint(epoch: int, model, tok, score: float, reason: str):
        """保存 checkpoint 到 checkpoints/epoch_N/"""
        epoch_dir = os.path.join(checkpoints_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        # Save LoRA adapter + tokenizer + trainer state (optimizer/scheduler/global_step/RNG)
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

        # 保存该 epoch 的元信息
        meta = {
            "epoch": epoch,
            "score": score,
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        file_utils.save_json(os.path.join(epoch_dir, "checkpoint_meta.json"), meta)

        print(f"[INFO] 💾 Saved checkpoint: {epoch_dir} ({reason}, score={score:.4f})")

    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model_name,
    #     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    #     device_map="auto" if torch.cuda.is_available() else None,
    # )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    from peft import PeftModel

    if args.resume_lora_dir:
        #  从已保存的 LoRA adapter 恢复
        model = PeftModel.from_pretrained(model, args.resume_lora_dir, is_trainable=True)
        print(f"[INFO] Resumed LoRA adapter from: {args.resume_lora_dir}")
    else:
        #  新建 LoRA
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_cfg)
    model.train()

    dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_collate_fn(tok))

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # total steps should be based on full planned epochs, not "remaining"
    total_steps_total = epochs * math.ceil(len(dl) / grad_accum)
    warmup_steps = max(10, total_steps_total // 20)

    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps_total,
    )

    # after optim/sched are created
    step = 0
    start_ep = 0

    if args.resume_lora_dir:
        # 1) load trainer_state.pt (optimizer/scheduler/rng/epoch/step)
        ckpt_state = load_checkpoint_lora(
            ckpt_dir=args.resume_lora_dir,
            model=model,
            optimizer=optim,
            scheduler=sched,
            scaler=None,
            map_location="cpu",
        )
        # 2) override start epoch / step from checkpoint (avoid manual args.resume_epoch)
        start_ep = int(ckpt_state.get("epoch", 0))
        step = int(ckpt_state.get("global_step", 0))
        print(f"[INFO]  Loaded trainer_state: epoch={start_ep}, global_step={step}")
    else:
        start_ep = 0
        step = 0


    # step = 0
    model.zero_grad(set_to_none=True)
    for ep in range(start_ep + 1, epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {ep}/{epochs}")
        print(f"{'=' * 60}")

        pbar = tqdm(dl, desc=f"Train epoch {ep}")
        for batch_i, batch in enumerate(pbar, start=1):
            # fix pad id
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

        # Dev evaluation
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
        )

        dev_report = {
            "epoch": ep,
            "dev": eval.build_eval_report_l1_and_l2_from_l1(dev_gold, dev_pred, args.l1tol2_mappings),
        }

        #  更新主 dev report（保持向后兼容）
        file_utils.save_json(args.report_out_dev, dev_report)

        score = get_nested(dev_report["dev"], ["level1", "overall", "micro_f1"], default=float("nan"))
        epoch_scores[ep] = score

        print(f"[INFO] Epoch {ep} score: {score:.4f}")

        #  决定是否保存 checkpoint
        should_save = False
        save_reason = ""

        # 1) 第一个 epoch：总是保存，且初始化为best（如果只有1个epoch）
        if ep == 1:
            should_save = True
            save_reason = "first epoch"

            #  如果只有1个epoch，第一个就是best
            if epochs == 1 or (isinstance(score, float) and not math.isnan(score)):
                if isinstance(score, float) and not math.isnan(score) and (best_epoch == -1 or score > best_score):
                    best_score = score
                    best_epoch = ep
                    # 第一个epoch也保存到best目录
                    # Save BEST with trainer state as well (so it can be resumed)
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
                    best_meta = {
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "metric": "level1.overall.micro_f1",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    file_utils.save_json(os.path.join(best_dir, "best_meta.json"), best_meta)
                    print(f"[INFO]  Initialized BEST with epoch 1 (score={best_score:.4f})")

        # 2) 新的 best（epoch > 1）：保存到 checkpoints/ 和 best/
        elif isinstance(score, float) and not math.isnan(score) and score > best_score:
            should_save = True
            save_reason = "new best"
            best_score = score
            best_epoch = ep

            # 同时保存到 best 目录
            # Save BEST with trainer state as well (so it can be resumed)
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

            best_meta = {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "metric": "level1.overall.micro_f1",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            file_utils.save_json(os.path.join(best_dir, "best_meta.json"), best_meta)
            print(f"[INFO]  NEW BEST! Saved to {best_dir} (epoch={best_epoch}, score={best_score:.4f})")

        # 3) 最后一个 epoch（且不是第一个）：总是保存
        elif ep == epochs and ep > 1:
            should_save = True
            save_reason = "last epoch"

        #  执行保存（如果需要）
        if should_save:
            _save_checkpoint(ep, model, tok, score, save_reason)

            # 同时保存该 epoch 的 dev report
            epoch_dir = os.path.join(checkpoints_dir, f"epoch_{ep}")
            epoch_report_path = os.path.join(epoch_dir, "dev_report.json")
            file_utils.save_json(epoch_report_path, dev_report)
        else:
            print(f"[INFO] Epoch {ep} not saved (current: {score:.4f}, best: {best_score:.4f} @ epoch {best_epoch})")

        model.train()

    #  保存训练摘要
    saved_epochs = []
    if 1 in epoch_scores:
        saved_epochs.append({"epoch": 1, "reason": "first", "score": epoch_scores[1]})
    if best_epoch > 0 and best_epoch not in [1, epochs]:
        saved_epochs.append({"epoch": best_epoch, "reason": "best", "score": best_score})
    if epochs > 1:
        saved_epochs.append({"epoch": epochs, "reason": "last", "score": epoch_scores.get(epochs, float('nan'))})

    training_summary = {
        "total_epochs": epochs,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metric": "level1.overall.micro_f1",
        "all_epoch_scores": {f"epoch_{k}": v for k, v in epoch_scores.items()},
        "saved_checkpoints": saved_epochs,
        "output_structure": {
            "best/": "Best performing checkpoint across all epochs",
            "checkpoints/epoch_N/": "Saved only for: first epoch, new best, and last epoch",
        }
    }
    file_utils.save_json(os.path.join(output_dir, "training_summary.json"), training_summary)

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"{'=' * 60}")
    print(f"Best epoch: {best_epoch} (score: {best_score:.4f})")
    print(f"Best model: {best_dir}")
    print(f"Saved checkpoints: {[ckpt['epoch'] for ckpt in saved_epochs]}")
    print(f"Checkpoints dir: {checkpoints_dir}")
    print(f"{'=' * 60}\n")

    return model, tok

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/unified_value_recognition.json")
    # args = parser.parse_args()
    # config = config.Config(args)

    # data
    ap.add_argument("--train_event", type=str, required=True)
    ap.add_argument("--train_hv", type=str, required=True)
    ap.add_argument("--dev_event", type=str, required=True)
    ap.add_argument("--dev_hv", type=str, required=True)
    ap.add_argument("--test_event", type=str, required=True)
    ap.add_argument("--test_hv", type=str, required=True)

    # run mode
    ap.add_argument("--mode", type=str, choices=["infer", "train_lora"], required=True)

    # group/model
    ap.add_argument("--group", type=str, choices=["G1", "G2", "G3"], required=True)
    ap.add_argument("--provider", type=str, default="https://api.deepseek.com/beta", help="for G1: openai/anthropic/gemini/deepseek")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--api_key", type=str, default="sk-YQAA")

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

    # ap.add_argument("--input_dir", type=str, default="n",
    #                 help="G1 experiment output")
    ap.add_argument("--pred_out", type=str, default="./outputs/pred_test.json")
    ap.add_argument("--report_out", type=str, default="./outputs/report_test.json")

    # GPU
    ap.add_argument("--gpu_ids", type=str, default="",
                    help="e.g. '0' or '0,1'. If set, will override CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--device_map", type=str, default="single", choices=["single", "auto"],
                    help="single=force single GPU; auto=transformers device_map='auto'")

    # debug / subset
    # sampling (debug)
    ap.add_argument("--sample_train_n", type=int, default=0,
                    help="if >0, sample N instances from TRAIN gold for quick pipeline test")
    ap.add_argument("--sample_dev_n", type=int, default=0,
                    help="if >0, sample N instances from DEV gold for quick pipeline test")
    ap.add_argument("--sample_test_n", type=int, default=0,
                    help="if >0, sample N instances from TEST gold for quick pipeline test")
    ap.add_argument("--sample_seed", type=int, default=42)


    # ===== CHANGE 4) 在 main() 里 argparse 增加这些参数 =====
    ap.add_argument("--adapter_dir", type=str, default="results/G3/outputs/ministral-3-8B-Instruct-2512/Ministral-3-8B-Instr_20260114_113808/checkpoints/epoch_6/",
                    help="For G3 infer: path to LoRA adapter dir (the out_dir from train_lora).")
    ap.add_argument("--load_dir", type=str, default="",
                    help="For infer: load full finetuned checkpoint from this dir (e.g., ECHV-LLAMA).")
    ap.add_argument("--merge_lora_infer", type=str, default="0",
                    help="For G3 infer with adapter_dir: merge LoRA into base before inference. 1/0")
    ap.add_argument("--save_merged_dir", type=str, default="",
                    help="For train_lora: additionally save merged full model to this dir.")
    ap.add_argument("--hf_token", type=str, default="",
                    help="Optional HuggingFace token (or set env HUGGINGFACE_HUB_TOKEN).")
    ap.add_argument("--hv1_label_dir", type=str, default="",
                    help="level-1 human value labels.")
    ap.add_argument("--hv_label_dir", type=str, default="",
                    help="level-1 human value labels.")
    ap.add_argument("--l1tol2_mappings", type=str, default="",
                    help="level-1 human value label to level-2 human value mappings")
    ap.add_argument("--canonical_prompt", type=str, default="",
                    help="traning prompt.")
    ap.add_argument("--canonical_system", type=str, default="",
                    help="traning system definition.")
    ap.add_argument("--infer_batch_size", type=int, default=4,
                    help="Batch size for local inference (G2/G3). Start from 2 if OOM.")
    ap.add_argument("--dump_sampled_dir", type=str, default="dataset/",
                    help="If set, after sampling, dump sampled train/dev/test event+hv json files into this dir.")
    # 新增：是否使用时间戳目录
    ap.add_argument("--use_timestamp", type=str, default="1",
                    help="If 1, create timestamped output dir to avoid overwriting previous runs. 0 to use fixed dir.")
    ap.add_argument("--run_name", type=str, default="",
                    help="Optional run name to include in output dir (e.g., 'baseline', 'exp1')")
    ap.add_argument("--g1_evaluate_only", type=str, default="n",
                    help="y: read the results then evaluate. n: pred then save into the file.")
    ap.add_argument("--echv_exp_path", type=str, default="n",
                    help="G1 experiment output")
    ap.add_argument("--g1_use_map_reduce", type=str2bool, default=False)
    ap.add_argument("--role1", type=str, default="developer",
                    help="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--resume_lora_dir", type=str, default="",
                    help="Resume LoRA training from a saved adapter checkpoint dir, e.g. out_dir/checkpoints/epoch_6")
    ap.add_argument("--resume_epoch", type=int, default=0,
                    help="If resuming, the last finished epoch number (e.g., 6). Training will continue from resume_epoch+1")

    DEBUG = True
    if DEBUG:
        sys.argv = [sys.argv[0]] + [
            "--mode", "infer",
            "--group", "G3",
            "--gpu_ids", "1",
            "--device_map", "auto",
            "--model_name", "mistralai/Ministral-3-8B-Instruct-2512",
            "--label_space", "L1",
            "--prompt_variant", "A",
            "--epochs", "12",
            "--lr", "2e-4",
            "--batch_size", "1",
            "--grad_accum", "8",
            "--max_len", "4096",
            "--max_new_tokens", "128",
            "--train_event",
            "dataset/training_dataset_total_formatted.json",
            "--train_hv",
            "dataset/train_gold_full.json",
            "--dev_event",
            "dataset/dev_dataset_total_formatted.json",
            "--dev_hv",
            "dataset/dev_gold_full.json",
            "--test_event",
            "dataset/test_dataset_total_formatted.json",
            "--test_hv",
            "dataset/deep_analysis/test/psedo_BCE_SCE_test_gold.json",
            "--canonical_prompt", "prompt/canonical_prompt.txt",
            "--canonical_system", "prompt/canonical_system.txt",
            "--hv1_label_dir", "dataset/hv/id2hv.json",
            "--hv_label_dir", "dataset/hv/values.json",
            "--l1tol2_mappings", "dataset/hv/l1tol2_id_mapping.json",
            "--sample_train_n", "99999",
            "--sample_dev_n", "99999",
            "--sample_test_n", "99999",
            "--sample_seed", "42",
            "--out_dir",
            "results/G3/outputs/mistralai/Ministral-3-8B-Instruct-2512_pseudo/debug_lora_ckpt",
            "--pred_out",
            "results/G3/outputs/mistralai/Ministral-3-8B-Instruct-2512_pseudo/debug_lora_ckpt/test_pred.json",
            "--report_out",
            "results/G3/outputs/mistralai/Ministral-3-8B-Instruct-2512_pseudo/debug_lora_ckpt/debug_report.json",
            "--pred_out_dev",
            "results/G3/outputs/mistralai/Ministral-3-8B-Instruct-2512_pseudo/debug_lora_ckpt/dev_test_pred.json",
            "--report_out_dev",
            "results/G3/outputs/mistralai/Ministral-3-8B-Instruct-2512_pseudo/debug_lora_ckpt/debug_dev_report.json",
            "--g1_evaluate_only", "y",
        ]
            # "--resume_lora_dir", "results/G3/outputs/mistralai/Ministral-3-8B-Instruct-2512/Llama-3.1-8B-Instruc_20260117_163307/checkpoints/epoch_4/",
            # "--resume_epoch", "4",
            # "--temperature_qwen", "0.6",

    args = ap.parse_args()
    hf_token = args.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN", None)

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    use_auto_map = (args.device_map == "auto")

    # ---- IMPORTANT: if resuming LoRA, reuse the previous run directory ----
    if args.resume_lora_dir:
        resume_run_dir = _resolve_run_dir_from_resume_path(args.resume_lora_dir)

        # force outputs to be written under the same run dir
        args.out_dir = resume_run_dir

        # keep pred/report inside the same run dir as well
        args.pred_out = os.path.join(args.out_dir, "test_pred.json")
        args.report_out = os.path.join(args.out_dir, "test_report.json")
        args.pred_out_dev = os.path.join(args.out_dir, "dev_pred.json")
        args.report_out_dev = os.path.join(args.out_dir, "dev_report.json")

        # (optional) you can also disable timestamp explicitly to avoid confusing logs
        args.use_timestamp = "0"

        print(f"\n{'=' * 70}")
        print(f" Resume detected. Reusing run directory:")
        print(f"  resume_from: {args.resume_lora_dir}")
        print(f"  run_dir    : {args.out_dir}")
        print(f"{'=' * 70}\n")

    # 生成带时间戳的输出目录
    if str2bool(args.use_timestamp):
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 构建目录名称
        if args.run_name:
            dir_suffix = f"{args.run_name}_{timestamp}"
        else:
            # 使用模型名的简短版本
            model_short = args.model_name.split('/')[-1][:20]  # 取最后部分，最多20字符
            dir_suffix = f"{model_short}_{timestamp}"

        # 重写所有输出路径
        base_out_dir = os.path.dirname(args.out_dir) or "outputs"
        args.out_dir = os.path.join(base_out_dir, dir_suffix)

        # 更新所有相关路径
        args.pred_out = os.path.join(args.out_dir, "test_pred.json")
        args.report_out = os.path.join(args.out_dir, "test_report.json")
        args.pred_out_dev = os.path.join(args.out_dir, "dev_pred.json")
        args.report_out_dev = os.path.join(args.out_dir, "dev_report.json")

        print(f"\n{'=' * 70}")
        print(f" Run ID: {dir_suffix}")
        print(f" Output directory: {args.out_dir}")
        print(f"{'=' * 70}\n")
    else:
        print(f"\n{'=' * 70}")
        print(f" Using fixed output directory: {args.out_dir}")
        print(f"  Warning: This will overwrite previous results!")
        print(f"{'=' * 70}\n")

    #  保存运行配置
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
    }

    # ---- load data ----
    train_event = file_utils.load_json(args.train_event)
    dev_event = file_utils.load_json(args.dev_event)
    test_event = file_utils.load_json(args.test_event)

    train_gold = file_utils.load_json(args.train_hv)
    dev_gold = file_utils.load_json(args.dev_hv)
    test_gold = file_utils.load_json(args.test_hv)

    # =============================================================================
    train_events_by_guid = data_utils.index_events_by_guid(train_event)
    dev_events_by_guid = data_utils.index_events_by_guid(dev_event)
    test_events_by_guid = data_utils.index_events_by_guid(test_event)

    train_gold = data_utils.list_to_gold_dict(train_gold)
    dev_gold = data_utils.list_to_gold_dict(dev_gold)
    test_gold = data_utils.list_to_pseudo_gold_dict(test_gold)

    # checkresult = check_guids(train_gold, dev_gold, test_gold)

    # ---------- DEBUG sampling ----------
    if args.sample_train_n > 0:
        train_gold = data_utils.sample_gold_instances(train_gold, args.sample_train_n, args.sample_seed)
        train_events_by_guid = data_utils.filter_events_by_gold(train_events_by_guid, train_gold)

    if args.sample_dev_n > 0:
        dev_gold = data_utils.sample_gold_instances(dev_gold, args.sample_dev_n, args.sample_seed + 1)
        dev_events_by_guid = data_utils.filter_events_by_gold(dev_events_by_guid, dev_gold)

    if args.sample_test_n > 0:
        test_gold = data_utils.sample_gold_instances(test_gold, args.sample_test_n, args.sample_seed + 2)
        test_events_by_guid = data_utils.filter_events_by_gold(test_events_by_guid, test_gold)
    # -----------------------------------

    hv_label_names_str = prompt_utils.get_hv1_mappings(args.hv1_label_dir)
    hv_label_description_str = prompt_utils.get_hv_description_dict(args.hv_label_dir)
    prompt_content = file_utils.python_file_to_json(args.canonical_prompt)
    prompt_system = file_utils.python_file_to_json(args.canonical_system)

    print("[INFO] instances:", len(train_gold), len(dev_gold), len(test_gold))

    # ========================================================================
    # d 是你的 dict 获取长文本测试 mapreduce是你的 dict
    # test_abcs = list(test_gold.items())
    # test_abcs_new = []
    # test_abc_keys_new = []
    # test_abcs_keys = list(test_gold.keys())
    # for test_abcs_key in test_abcs_keys:
    #     guid = test_abcs_key[0]
    #     unit_level = test_abcs_key[1]
    #     e = test_events_by_guid.get(guid)
    #     article_content = e["content"]
    #     test_content = prompt_content + article_content
    #     if unit_level == "article":
    #         prompt_tokens = data_utils._count_tokens_api(test_content, "gpt-5.2")
    #         if prompt_tokens > 4096:
    #             test_abc_keys_new.append(test_abcs_key)
    # for test_abc in test_abcs:
    #     key = test_abc[0]
    #     if key in test_abc_keys_new:
    #         test_abcs_new.append(test_abc)

    # for test_abc_key_new in test_abc_keys_new:
    #     test_abcs_new.append(test_gold[test_abc_key_new])
    # test_gold = dict(test_abcs_new[:10])
    # ========================================================================

    # test_gold = dict(list(test_gold.items())[:5])
    if args.mode == "infer":
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
                    echv_exp_path = args.echv_exp_path,
                    g1_use_map_reduce = True,
                    role1 = args.role1,
                    deepseek_base_url=args.provider
                )
            else:
                folder_names = list(range(50))
                # folder_names = [0]  # [0, 1, 2, ..., 50]
                all_records: List[Dict[str, Any]] = []
                for folder_name in folder_names:
                    tmp_result_path = args.echv_exp_path + args.group + "/output/" + args.model_name + "/" + str(folder_name) + "/output_results.json"
                    data = data_utils.load_json(tmp_result_path)
                    if not isinstance(data, list):
                        print(f"[WARN] {fp} is not a list, skip.")
                        continue
                    all_records.extend(data)

                print(f" Total records: {len(all_records)}")

                pred = data_utils.build_pred_from_records(all_records)
                print(f" Unique InstanceId (pred size): {len(pred)}")
                empty_cnt = data_utils.count_empty_pred_instances(pred)
                print(f"[PRED] empty (aligned+contradictory both empty): {empty_cnt} / {len(pred)}")
                # serializable = pred_to_serializable(pred)
                # save_json(args.pred_out, serializable)
        else:
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

        if args.g1_evaluate_only == "y":
            # save raw pred (aligned/contra sets)
            serializable = []
            for iid, d in pred.items():
                guid, unit_level, unit_id, actor = iid
                serializable.append({
                    "guid": guid,
                    "unit_level": unit_level,
                    "unit_id": unit_id,
                    "actor": actor,
                    "aligned_with_human_values": sorted(list(d["aligned"])),
                    "contradictory_to_human_values": sorted(list(d["contradictory"])),
                })
            file_utils.save_json(args.pred_out, serializable)

            report = eval.build_eval_report_l1_and_l2_from_l1(
                gold_l1=test_gold,
                pred_l1=pred,
                l1tol2_mapping_path=args.l1tol2_mappings,
            )

            file_utils.save_json(args.report_out, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))

            print(f"\n{'=' * 70}")
            print(f" Inference Complete!")
            print(f"📊 Test predictions: {args.pred_out}")
            print(f"📊 Test report: {args.report_out}")
            print(f"{'=' * 70}\n")

    else:
        # LoRA training (G3)
        assert args.group == "G3", "train_lora is only for G3"
        assert torch is not None, "torch not installed"
        assert AutoTokenizer is not None and AutoModelForCausalLM is not None, "transformers not installed"

        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        # Build SFT examples from TRAIN gold
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
            group=args.group
        )
        print("[INFO] SFT examples:", len(train_examples))

        train_ds = SFTDataset(train_examples, tok, max_len=args.max_len)

        def debug_check_mask(train_ds, tok, n=3):
            import random
            for _ in range(n):
                item = train_ds[random.randrange(len(train_ds))]
                labels = item["labels"].tolist()
                input_ids = item["input_ids"].tolist()

                first_sup = next((i for i, x in enumerate(labels) if x != -100), None)
                print("first supervised pos:", first_sup, "total_len:", len(labels))
                if first_sup is not None:
                    print("prompt_tail:",
                          tok.decode(input_ids[max(0, first_sup - 80):first_sup], skip_special_tokens=True))
                    print("target_head:", tok.decode(input_ids[first_sup:first_sup + 120], skip_special_tokens=True))
                print("-" * 60)

        # 用法：
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
            max_new_tokens_eval=args.max_new_tokens
        )
        # ===== CHANGE 7) 在 train_lora 分支训练完后（你现在 train_lora_sft 里已经 save_pretrained 了）=====
        # 你在 main() 的 train_lora 分支里拿到 model,tok 后，加：
        maybe_save_merged_full_model(model, tok, args.save_merged_dir)

        # final test eval
        # --------- AFTER TRAIN: load best for test (if exists) ----------
        best_adapter_dir = os.path.join(args.out_dir, "best")
        use_adapter_dir = best_adapter_dir if os.path.isdir(best_adapter_dir) else args.out_dir

        model_best, tok_best = load_model_for_infer(
            group="G3",
            model_name=args.model_name,
            use_auto_map=use_auto_map,
            hf_token=hf_token,
            adapter_dir=use_adapter_dir,  #  best 优先
            load_dir="",
            merge_lora_infer=False,
        )

        # final test eval + save pred
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

        # 保存 test 预测内容（和 infer 分支一致）
        serializable = []
        for iid, d in test_pred.items():
            guid, unit_level, unit_id, actor = iid
            serializable.append({
                "guid": guid,
                "unit_level": unit_level,
                "unit_id": unit_id,
                "actor": actor,
                "aligned_with_human_values": sorted(list(d["aligned"])),
                "contradictory_to_human_values": sorted(list(d["contradictory"])),
            })
        file_utils.save_json(args.pred_out, serializable)
        print(f"[INFO] saved test predictions to: {args.pred_out}")

        # report = eval.build_eval_views(test_gold, test_pred, label_space=args.label_space)
        report = eval.build_eval_report_l1_and_l2_from_l1(
            gold_l1=test_gold,
            pred_l1=test_pred,
            l1tol2_mapping_path=args.l1tol2_mappings
        )
        file_utils.save_json(args.report_out, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))

        print(f"\n{'=' * 70}")
        print(f" Training & Evaluation Complete!")
        print(f" All outputs saved to: {args.out_dir}")
        print(f" Test predictions: {args.pred_out}")
        print(f" Test report: {args.report_out}")
        print(f" Best model: {os.path.join(args.out_dir, 'best')}")
        print(f"{'=' * 70}\n")

    file_utils.save_json(os.path.join(args.out_dir, "run_config.json"), run_config)

if __name__ == "__main__":
    main()