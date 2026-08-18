"""Data loading, formatting, payload construction, and prediction utilities.

This module contains shared helpers for converting event and human-value label
files into instance-keyed dictionaries, building model input payloads for each
unit level, sampling balanced subsets, and normalizing model predictions.
"""

import os
import json
import sys
import time
import numpy as np
from typing import Any, Dict, List, Tuple, Set, Optional, Union, Iterable
import random
from collections import defaultdict, Counter
import threading
from google import genai
from openai import OpenAI
from utils import llm_utils
from utils import file_utils
from utils import preprocess_utils
from utils import prompt_utils
from utils import token_utils
from utils import data_utils
from utils import statistics_utils
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import glob
InstanceId = Tuple[str, str, str, str]  # (guid, unit_level, unit_id, actor)
import math
try:
    import tiktoken
except Exception:
    tiktoken = None


def find_subfile_by_guid(guid_key, results_model_subfile):
    for result_model_subfile in results_model_subfile:
        guid_model = result_model_subfile["guid"]
        if guid_key == guid_model:
            return result_model_subfile

def finditem(guid, data_file):
    # if data_file == None:
    #     return None
    for item in data_file:
        temp_guid = item["guid"]
        if guid == temp_guid:
            return item
    print("not item is found!")
    return {}
def finditems_by_guid(guid, data_file):
    results = []
    for item in data_file:
        temp_guid = item["guid"]
        if guid == temp_guid:
            results.append(item)
    print("not item is found!")
    return results

def find_item_by_key(comp_id, search_key, items):
    for item in items:
        temp_id = item[search_key]
        if str(comp_id) == str(temp_id):
            return item
    return {}

def find_item_by_keys(comp_id, search_key, items):
    for item in items:
        temp_id = item[search_key]
        if comp_id == temp_id:
            return item
    return {}

def finedsubevent(search_subevent_id, subevent_id, subevents_hvs):
    # if subevents_hvs:
    #     return {}
    for subevent_hvs in subevents_hvs:
        if search_subevent_id not in subevent_hvs:
            search_subevent_id = "id"
            # return {}
        temp_subevent_id = subevent_hvs[search_subevent_id]
        if subevent_id == temp_subevent_id:
            return subevent_hvs
    print("not subevent is found!")
    return {}

def fined_behavior_chains(behavior_chain_id, behavior_chains_hvs):
    # if subevents_hvs:
    #     return {}
    for behavior_chain_hvs in behavior_chains_hvs:
        temp_behavior_ids_chains = behavior_chain_hvs["behavior_ids_chains"]
        if behavior_chain_id == temp_behavior_ids_chains:
            return behavior_chain_hvs
    print("not behavior_chain_hvs is found!")
    return {}

def fined_story_narr(story_narr_id, story_narr_hvs):
    for story_narr_hv in story_narr_hvs:
        # if story_narr_id not in story_narr_hv.keys():
        #     story_narr_id = "id"
        #     print("story_narr_id not exist!")
        temp_story_narr_id = story_narr_hv["story_narrative_id"]
        if story_narr_id == temp_story_narr_id:
            return story_narr_hv
    print("not story_narr_hv is found!")
    return {}

def getvalue(key, dict_data):
    if key in dict_data:
        return dict_data[key]
    else:
        print("key not in dict_data")
        return {}

def find_hv_behavior_ids(behavior_ids_chains_key, behavior_chain_human_values):
    for specific_item in specific_human_values:
        if search_key == specific_item[behavior_ids_chains_key]:
            return specific_item
    return None

def find_story_narrative(search_key, story_narratives):
    for story_narrative in story_narratives:
        # if "Title" in story_narrative.keys():
        #     if search_key == story_narrative["id"]:
        #         return story_narrative
        # else:
        for story_narrative_title, story_narrative_detail in story_narrative.items():
            # if isinstance(story_narrative_detail, dict):
            #     if isinstance(story_narrative_detail["id"], int):
            #         print("story_narrative_detail id is integer")
            # else:
            #     print("story_narrative_detail is not dict!")
            # if isinstance(search_key, int):
            #     print("search_key is integer")
            # if isinstance(story_narrative_detail, str):
            #     print("story_narrative_detail is str!")
            # if "id" not in story_narrative_detail.keys():
            #     print("id not exist in story_narrative_detail!")
            if "id" not in story_narrative_detail:
                print("id not exist in story_narrative_detail!")
            if search_key == story_narrative_detail["id"]:
                return story_narrative
    return None

def find_related_subevent_ids(story_narrative):
    if "related_subevent_ids" in story_narrative:
        return story_narrative["related_subevent_ids"]
    else:
        related_subevent_ids = next(iter(story_narrative.values()))["related_subevent_ids"]
        # for story_narrative_title, story_narrative_detail in story_narrative.items():
        return related_subevent_ids
    return None

def get_behaviors_by_ids(behavior_chains, behavior_ids_chains):
    """
    behavior_chains: List[Dict[str, List[behavior_dict]]]
    behavior_ids_chains: List[str]
    """
    result = defaultdict(list)
    target_ids = set(behavior_ids_chains)

    for chain in behavior_chains:
        for title, behaviors in chain.items():
            for b in behaviors:
                if b.get("id") in target_ids:
                    result[title].append(b)

    # Sort by order, converting strings to int so '10' does not come before '2'
    for title in result:
        result[title] = sorted(
            result[title],
            key=lambda x: int(x.get("order", 0))
        )

    return dict(result)

def find_behavior_related_subevent_ids(behavior_chain, chain_type):
    related_subevent_ids = []
    if chain_type == "behavior_chain":
        behavior_chains = [behavior_chain]
    else:
        behavior_chains = behavior_chain
    for behavior_chain_item in behavior_chains:
        for behavior_chain_title, behaviors in behavior_chain_item.items():
            for behavior in behaviors:
                if "subevent_id" not in behavior:
                    print("subevent_id not in behavior!")
                subevent_id = behavior["subevent_id"]
                if subevent_id not in related_subevent_ids:
                    related_subevent_ids.append(subevent_id)
    return related_subevent_ids

def find_subevents_with_id(related_subevent_ids, subevents):
    related_subevents = []
    for subevent in subevents:
        new_subevent = {}
        if subevent["id"] in related_subevent_ids:
            new_subevent["subevent_id"] = subevent["id"]
            new_subevent["subevent"] = subevent["subevent"]
            related_subevents.append(new_subevent)
    return related_subevents

def find_subevent_id(subevent_human_values):
    for subeventid, subevent_value in subevent_human_values.items():
        if subeventid == "subevent_id":
            return subevent_value
    return None


def find_subevent(subevent_id, subevents):
    for subevent in subevents:
        if subevent_id == subevent["id"]:
            return subevent
    return None

def find_story_narrative(search_key, story_narratives):
    for story_narrative in story_narratives:
        # if "Title" in story_narrative.keys():
        #     if search_key == story_narrative["id"]:
        #         return story_narrative
        # else:
        for story_narrative_title, story_narrative_detail in story_narrative.items():
            # if isinstance(story_narrative_detail, dict):
            #     if isinstance(story_narrative_detail["id"], int):
            #         print("story_narrative_detail id is integer")
            # else:
            #     print("story_narrative_detail is not dict!")
            # if isinstance(search_key, int):
            #     print("search_key is integer")
            # if isinstance(story_narrative_detail, str):
            #     print("story_narrative_detail is str!")
            # if "id" not in story_narrative_detail.keys():
            #     print("id not exist in story_narrative_detail!")
            if "id" not in story_narrative_detail:
                print("id not exist in story_narrative_detail!")
            if search_key == story_narrative_detail["id"]:
                return story_narrative
    return None

def find_related_subevent_ids(story_narrative):
    if "related_subevent_ids" in story_narrative:
        return story_narrative["related_subevent_ids"]
    else:
        related_subevent_ids = next(iter(story_narrative.values()))["related_subevent_ids"]
        # for story_narrative_title, story_narrative_detail in story_narrative.items():
        return related_subevent_ids
    return None

def find_behavior_related_subevent_ids(behavior_chain, chain_type):
    related_subevent_ids = []
    if chain_type == "behavior_chain":
        behavior_chains = [behavior_chain]
    else:
        behavior_chains = behavior_chain
    for behavior_chain_item in behavior_chains:
        for behavior_chain_title, behaviors in behavior_chain_item.items():
            for behavior in behaviors:
                if "subevent_id" not in behavior:
                    print("subevent_id not in behavior!")
                subevent_id = behavior["subevent_id"]
                if subevent_id not in related_subevent_ids:
                    related_subevent_ids.append(subevent_id)
    return related_subevent_ids

def find_subevent(subevent_id, subevents):
    for subevent in subevents:
        if subevent_id == subevent["id"]:
            return subevent
    return None

def find_sentence_ids(subevent_id, sent_subevent_mapping_results):
    for sent_subevent_mapping_result in sent_subevent_mapping_results:
        if subevent_id == sent_subevent_mapping_result["id"]:
            return sent_subevent_mapping_result["sentence_ids"]
    return []

def find_related_mappings(related_subevent_ids, mapping_results):
    related_mappings = []
    for mapping_result in mapping_results:
        subevent_id = mapping_result["subevent_id"]
        if str(subevent_id) in related_subevent_ids:
            related_mappings.append(mapping_result)
    return related_mappings

def compute_actor_importance(related_mappings, w1=0.6, w2=0.4, normalize=True, smooth=True):
    """
    Compute each actor's importance score with smoothed normalization
    """
    relevance_ranges = {
        "yes": (0.8, 1.0),
        "partial": (0.4, 0.7),
        "no": (0.1, 0.2),
    }

    def map_relevance_value(relevance, conf):
        relevance = relevance.lower()
        low, high = relevance_ranges.get(relevance, (0.0, 0.0))
        return low + (high - low) * conf

    actor_scores = {}

    # Compute each actor's accumulated score
    for sent in related_mappings:
        if "actors" not in sent:
            print("test")
        for a in sent["actors"]:
            actor = a["actor"]
            conf = float(a["confidence"])
            rel = a["relevance"].lower()
            rel_weight = map_relevance_value(rel, conf)
            score = rel_weight * w1 + conf * w2
            actor_scores[actor] = actor_scores.get(actor, 0.0) + score

    # Smoothed normalization
    if normalize and actor_scores:
        vals = np.array(list(actor_scores.values()))
        min_v, max_v = vals.min(), vals.max()

        if max_v - min_v > 1e-8:
            if smooth:
                eps = 0.1  # buffer to keep bounds away from 0 and 1
                for k in actor_scores:
                    norm = (actor_scores[k] - min_v) / (max_v - min_v)
                    actor_scores[k] = eps + (1 - 2*eps) * norm
            else:
                for k in actor_scores:
                    actor_scores[k] = (actor_scores[k] - min_v) / (max_v - min_v)
        else:
            for k in actor_scores:
                actor_scores[k] = 0.5  # Assign a neutral score when all values are equal
    return actor_scores

def compute_hv_importance(impact_result, w1=0.6, w2=0.4, normalize=True, smooth=True):
    """
    Compute the importance of each (actor, human_value_label) from removal_impact_decision and confidence.

    Args:
        data: list[dict]
            Each element format:
            {
                "actor": str,
                "human_value_label": str,
                "removal_impact_decision": "yes|partial|no",
                "confidence": "9.00e-01"
            }
        w1, w2: float
            Weighting ratio between relevance and confidence
        normalize: bool
            Whether to normalize to [0, 1]
        smooth: bool
            Whether to use soft normalization to avoid extreme 0/1 values
    Returns:
        dict { (actor, human_value_label): importance_score }
    """

    # Define the semantic interval for each decision
    decision_ranges = {
        "yes": (0.8, 1.0),
        "partial": (0.4, 0.7),
        "no": (0.1, 0.2),
    }

    def map_decision_value(decision, conf):
        """Map decision and confidence to a continuous weight"""
        decision = decision.lower()
        low, high = decision_ranges.get(decision, (0.0, 0.0))
        return low + (high - low) * conf

    value_scores = {}

    # Iterate over all records
    for item in impact_result:
        actor = item["actor"]
        value = item["human_value_label"]
        key = (actor, value)

        conf = float(item["confidence"])
        dec = item["removal_impact_decision"].lower()
        dec_weight = map_decision_value(dec, conf)

        score = dec_weight * w1 + conf * w2
        value_scores[key] = value_scores.get(key, 0.0) + score

    # Smoothed normalization
    if normalize and value_scores:
        vals = np.array(list(value_scores.values()))
        min_v, max_v = vals.min(), vals.max()

        if max_v - min_v > 1e-8:
            if smooth:
                eps = 0.1  # control bounds to avoid extreme 0/1 values
                for k in value_scores:
                    norm = (value_scores[k] - min_v) / (max_v - min_v)
                    value_scores[k] = eps + (1 - 2 * eps) * norm
            else:
                for k in value_scores:
                    value_scores[k] = (value_scores[k] - min_v) / (max_v - min_v)
        else:
            for k in value_scores:
                value_scores[k] = 0.5  # Assign a neutral score when all values are equal

    return value_scores

def find_hv_specific_id(specifc_type, specific_human_values):
    for specific_id, specific_value in specific_human_values.items():
        if specific_id == specifc_type:
            return specific_value
    return None

def find_hv_specific_id(specifc_type, specific_human_values):
    for specific_id, specific_value in specific_human_values.items():
        if specific_id == specifc_type:
            return specific_value
    return None

def _get_article_title_content(e: Dict[str, Any]) -> Tuple[str, str]:
    # best effort, since schema may vary
    title = e["title"]
    content = e["content"]
    return title, content

def _find_subevent_text(e: Dict[str, Any], subevent_id: str) -> str:
    subs = e.get("subevents")
    if isinstance(subs, list):
        for se in subs:
            if not isinstance(se, dict):
                continue
            sid = str(se.get("id", "")).strip()
            if sid == str(subevent_id).strip():
                for k in ["subevent"]:
                    if isinstance(se.get(k), str) and se[k].strip():
                        return se[k].strip()
    return ""

def _find_subevent_info(e: Dict[str, Any], subevent_id: str) -> Tuple[str, List[str]]:
    """
    Find a subevent by id in e["subevents"] and return:
      (subevent_text, sentence_ids)

    Expected subevent format:
    {
      "id": "0",
      "subevent": "...",
      "sentence_ids": ["0", ...]
    }
    """
    subs = e.get("subevents")
    if isinstance(subs, list):
        target_id = str(subevent_id).strip()
        for se in subs:
            if not isinstance(se, dict):
                continue

            sid = str(se.get("id", "")).strip()
            if sid != target_id:
                continue

            # text
            text = ""
            if isinstance(se.get("subevent"), str) and se["subevent"].strip():
                text = se["subevent"].strip()

            # sentence_ids
            sent_ids: List[str] = []
            raw_ids = se.get("sentence_ids", [])
            if isinstance(raw_ids, list):
                sent_ids = [str(x).strip() for x in raw_ids if str(x).strip()]
            elif isinstance(raw_ids, (str, int)):
                # fallback if sentence_ids accidentally stored as single value
                s = str(raw_ids).strip()
                sent_ids = [s] if s else []

            return text, sent_ids

    return "", []


def _iter_behaviors(e: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """e["behaviors"] = [ {title: [ {...}, {...} ]}, ... ]"""
    raw = e.get("behaviors", [])
    if not isinstance(raw, list):
        return
    for item in raw:
        if isinstance(item, dict) and len(item) == 1:
            title, arr = next(iter(item.items()))
            if isinstance(title, str) and isinstance(arr, list):
                for b in arr:
                    if isinstance(b, dict):
                        yield title, b

def _uniq_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

from typing import Any, Dict, List, Tuple

def _behavior_chain_unit_text_and_related_cgt(
    e: Dict[str, Any],
    behavior_ids: List[str],   # Keep the signature for external compatibility, but do not use this argument
    related_ids: List[str],
) -> Tuple[str, str]:
    """
    Concatenate only subevent text selected by related_ids.
    Return: (unit_title, unit_text)

    unit_text format, single line:
      "Subevents: [<subevent_text_1>; <subevent_text_2>; ...]"
    If related_ids is empty:
      "Subevents: []"
    """
    sub_texts: List[str] = []
    for sid in related_ids:
        se_txt = _find_subevent_text(e, sid)
        if isinstance(se_txt, str):
            se_txt = se_txt.strip()
        else:
            se_txt = str(se_txt).strip()
        sub_texts.append(se_txt)

    sub_str = "; ".join([t for t in sub_texts if t])

    if sub_str:
        unit_text = f"Subevents: [{sub_str}]"
    else:
        unit_text = "Subevents: []"

    unit_title = ""
    return unit_title, unit_text

def _behavior_chain_unit_text_and_related(
    e: Dict[str, Any],
    behavior_ids: List[str],
) -> Tuple[str, str, List[str]]:
    """
    Input: behavior_ids, e.g. ["0", "1", "2", "3"]
    Return: (unit_title, unit_text, related_subevent_ids)

    unit_text format, one behavior per line:
      "<behavior_text>: [<subevent_text_1>; <subevent_text_2>; ...]"
    Lines are joined with "\n"

    related_ids:
      - Add subevent_id to related_ids whenever it exists and is not None, converted to str
      - Skip missing or None subevent_id values
    """
    id_set: Set[str] = {str(x) for x in behavior_ids}

    titles: List[str] = []
    lines: List[str] = []
    related: List[str] = []

    for title, b in _iter_behaviors(e):
        bid = b.get("id")
        if bid is None or str(bid) not in id_set:
            continue

        if title:
            titles.append(title)

        behavior_txt = b.get("behavior")
        if not isinstance(behavior_txt, str) or not behavior_txt.strip():
            behavior_txt = ""
        else:
            behavior_txt = behavior_txt.strip()

        # subevent_id -> subevent text list
        sub_texts: List[str] = []
        if "subevent_id" in b and b["subevent_id"] is not None:
            sid = str(b["subevent_id"])
            related.append(sid)

            se_txt = _find_subevent_text(e, sid)
            if isinstance(se_txt, str) and se_txt.strip():
                sub_texts.append(se_txt.strip())

        # Assemble in the expected format: behavior: [subevents]
        if sub_texts:
            sub_str = "; ".join(sub_texts)
            lines.append(f"{behavior_txt}: [{sub_str}]")
        else:
            lines.append(f"{behavior_txt}: []")

    unit_title = " / ".join(_uniq_keep_order([t for t in titles if t]))
    unit_text = "\n".join([ln for ln in lines if isinstance(ln, str) and ln.strip()])
    related_ids = _uniq_keep_order([r for r in related if r])

    return unit_title, unit_text, related_ids

def _story_unit_text_and_related_cgt(
    e: Dict[str, Any],
    st_id: str,               # Keep the signature for compatibility; unused here
    related_ids: List[str]
) -> Tuple[str, str]:
    sub_texts: List[str] = []
    for sid in related_ids:
        txt = _find_subevent_text(e, sid)
        if isinstance(txt, str):
            txt = txt.strip()
        else:
            txt = str(txt).strip()
        if txt:
            sub_texts.append(txt)

    if sub_texts:
        unit_text = "Subevents: [" + "; ".join(sub_texts) + "]"
    else:
        unit_text = "Subevents: []"

    return "", unit_text

def _story_unit_text_and_related(e: Dict[str, Any], st_id: str) -> Tuple[str, str, List[str]]:
    st = _find_story_narrative(e, st_id)
    if not st:
        return "", "", []
    else:
        # Preserve title, description, and related fields
        unit_title = str(st.get("title", "")).strip()
        desc = str(st.get("description", "")).strip()

        related = st.get("related_subevent_ids", []) or []
        related = [str(x) for x in related] if isinstance(related, list) else []

        #  Only change unit_text: Title: description \n Subevents \n <subevent texts...>
        sub_texts: List[str] = []
        for sid in related:
            txt = _find_subevent_text(e, sid)
            if isinstance(txt, str) and txt.strip():
                sub_texts.append(txt.strip())

        if sub_texts:
            unit_text = desc.strip() + "\n" + "Subevents:" + "\n" + "\n".join(sub_texts)
        else:
            unit_text = desc.strip() + "\n" + "Subevents:"

        return unit_title, unit_text, related

def _get_linked_subevents(e: Dict[str, Any], related_ids: List[str]) -> List[Dict[str, str]]:
    out = []
    for sid in related_ids:
        txt = _find_subevent_text(e, sid)
        if txt:
            out.append({"subevent_id": str(sid), "text": txt})
    return out

def _find_behavior_chain(e: Dict[str, Any], bc_id: str) -> Optional[Dict[str, Any]]:
    # you may store behavior chains under different keys
    behaviors = e.get("behaviors")
    behaviors = get_behaviors_by_ids(behaviors, bc_id)

    if behaviors:
        return behaviors
    else:
        return None

def _find_story_narrative(e: Dict[str, Any], st_id: str) -> Optional[Dict[str, Any]]:
    sns = e.get("story_narratives")
    if isinstance(sns, list):
        for st in sns:
            if not isinstance(st, dict):
                continue
            _id = st.get("id", None)
            if _id is not None and str(_id).strip() == str(st_id).strip():
                return st
    return None

def get_sentence_ids_by_subevent_ids(
    e: Dict[str, Any],
    subevent_ids: Union[List[Union[str, int]], Set[Union[str, int]]]
) -> List[str]:
    """
    Return a de-duplicated sentence_id list for the given subevent_ids.
    - De-duplicates sentence_ids
    - Sorts sentence_ids numerically when possible, else lexicographically
    """
    # normalize target subevent ids
    target_ids: Set[str] = {str(x).strip() for x in subevent_ids if x is not None and str(x).strip()}
    if not target_ids:
        return []

    subs = e.get("subevents", [])
    if not isinstance(subs, list):
        return []

    # collect & de-duplicate
    sent_set: Set[str] = set()
    for se in subs:
        if not isinstance(se, dict):
            continue
        sid = str(se.get("id", "")).strip()
        if sid not in target_ids:
            continue

        raw = se.get("sentence_ids", [])
        if isinstance(raw, list):
            for x in raw:
                s = str(x).strip()
                if s:
                    sent_set.add(s)
        elif raw is not None:
            s = str(raw).strip()
            if s:
                sent_set.add(s)

    # sort (numeric if possible)
    def _sort_key(x: str):
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    return sorted(sent_set, key=_sort_key)

def _get_actor_by_id(actor_id, actors):
    flipped_dict = {v: k for d in actors for k, v in d.items()}
    actor = flipped_dict[actor_id]
    return actor

from typing import Any, Dict, List, Union, Set

def get_sentence_texts_by_ids(
    e: Dict[str, Any],
    sentence_ids: Union[List[Union[str, int]], Set[Union[str, int]]],
    *,
    include_heading: bool = False
) -> List[str]:
    """
    Given an event/article dict `e` with:
      e["sentences"] = [{"type": "...", "sentence_id": 0, "text": "..."}, ...]
    and a list/set of sentence_ids, return the corresponding sentence texts in the
    order of `sentence_ids` (deduplicated, preserving first occurrence order).

    - By default, only returns items with type=="sentence".
    - Set include_heading=True to also allow type=="heading" when it has a sentence_id
      (some datasets may not give headings an id).
    """
    # normalize ids, preserve input order (dedup)
    ordered_ids: List[str] = []
    seen: Set[str] = set()
    for x in sentence_ids:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            ordered_ids.append(s)

    if not ordered_ids:
        return []

    sents = e.get("sentences", [])
    if not isinstance(sents, list):
        return []

    # build lookup: id(str) -> text
    id2text: Dict[str, str] = {}
    for obj in sents:
        if not isinstance(obj, dict):
            continue
        typ = obj.get("type")
        if typ == "sentence" or (include_heading and typ == "heading"):
            sid = obj.get("sentence_id", None)
            if sid is None:
                continue
            key = str(sid).strip()
            txt = obj.get("text", "")
            if isinstance(txt, str) and txt.strip():
                # keep first occurrence
                id2text.setdefault(key, txt.strip())

    # return texts following requested id order (skip missing)
    return [id2text[i] for i in ordered_ids if i in id2text]

def join_sentence_texts(
    texts: List[str],
    *,
    sep: str = "\n",
    prefix_with_index: bool = False
) -> str:
    """
    Merge a list of sentence texts into one string.

    - sep: separator between sentences (default newline)
    - prefix_with_index: if True, output like "0. sentence..."
    """
    cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned:
        return ""

    if prefix_with_index:
        cleaned = [f"{i}. {t}" for i, t in enumerate(cleaned)]

    return sep.join(cleaned)

def get_all_subevent_sentence_ids(e: Dict[str, Any]) -> List[str]:
    """
    Collect all sentence_ids from e["subevents"], de-duplicate, and return
    a sorted list (numeric sort if possible).
    """
    subs = e.get("subevents", [])
    if not isinstance(subs, list):
        return []

    sent_set: Set[str] = set()
    for se in subs:
        if not isinstance(se, dict):
            continue
        raw = se.get("sentence_ids", [])
        if isinstance(raw, list):
            for x in raw:
                s = str(x).strip()
                if s:
                    sent_set.add(s)
        elif isinstance(raw, (str, int)):
            s = str(raw).strip()
            if s:
                sent_set.add(s)

    # numeric sort if possible
    try:
        return sorted(sent_set, key=lambda x: int(x))
    except ValueError:
        return sorted(sent_set)

# ----------------------------
# Build instance-level gold labels from hv rows
#   instance_id = (guid, unit_level, unit_id, actor)
#   labels stored as sets for aligned / contradictory
# ----------------------------
InstanceId = Tuple[str, str, str, str]  # guid, unit_level, unit_id, actor
# ----------------------------
# Label universes (IDs as strings for JSON output)
# ----------------------------
L1_IDS = [str(i) for i in range(54)]
L2_IDS = [str(i) for i in range(20)]

from typing import Any, Dict, List, Set, Tuple, Union

def get_neighbor_context_sentences(
    data: Dict[str, Any],
    target_sentence_ids: List[Union[str, int]],
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Return:
      (out_with_ids, out_texts)
    """
    sents = data.get("sentences", [])
    if not isinstance(sents, list):
        return [], []   #  Fix: always return two values

    # Build map: sentence_id(str) -> text, only for type == "sentence"
    id2text: Dict[str, str] = {}
    max_id = -1
    for x in sents:
        if not isinstance(x, dict):
            continue
        if x.get("type") != "sentence":
            continue
        sid = x.get("sentence_id", None)
        if sid is None:
            continue
        try:
            sid_int = int(sid)
        except (TypeError, ValueError):
            continue
        txt = x.get("text", "")
        if not isinstance(txt, str):
            txt = str(txt)
        sid_str = str(sid_int)
        id2text[sid_str] = txt
        if sid_int > max_id:
            max_id = sid_int

    if max_id < 0:
        return [], []   # Fix.

    # normalize target set to int for boundary checks + membership
    target_set: Set[int] = set()
    for t in target_sentence_ids:
        try:
            target_set.add(int(str(t).strip()))
        except (TypeError, ValueError):
            continue

    neighbor_ids: Set[int] = set()
    for tid in target_set:
        for nb in (tid - 1, tid + 1):
            if nb < 0 or nb > max_id:
                continue
            if nb in target_set:
                continue
            neighbor_ids.add(nb)

    # Keep only those that exist AND are type == sentence (via id2text)
    out: List[Dict[str, str]] = []
    out_text: List[str] = []
    for nb in sorted(neighbor_ids):
        nb_str = str(nb)
        if nb_str in id2text:
            out.append({"sentence_id": nb_str, "sentence": id2text[nb_str]})
            out_text.append(id2text[nb_str])

    return out, out_text

def build_input_payload(
    e: Dict[str, Any],
    iid: InstanceId,
    label_space: str,
    hv_label: str,
    hv_label_desc: str
) -> Dict[str, Any]:

    guid, unit_level, unit_id, actor_id = iid
    article_title, content = _get_article_title_content(e)

    sentence_ids = []
    # evidence based article hv recognition
    # if unit_level == "article":
    #     unit_text = ""
    #     unit_title = ""
    #     linked = []
    #     sentence_ids = get_all_subevent_sentence_ids(e)
    # raw article-based hv recognition
    if unit_level == "article":
        unit_text = content
        unit_title = article_title
        linked = []
        sentence_ids = []
    elif unit_level == "subevent":
        unit_text, sentence_ids = _find_subevent_info(e, unit_id)
        unit_title = ""
        linked = []
    elif unit_level == "behavior_chain":
        unit_title, unit_text, related_ids = _behavior_chain_unit_text_and_related(e, unit_id)
        linked = _get_linked_subevents(e, related_ids)
        sentence_ids = get_sentence_ids_by_subevent_ids(e, related_ids)
    elif unit_level == "story_narrative":
        unit_title, unit_text, related_ids = _story_unit_text_and_related(e, unit_id)
        linked = _get_linked_subevents(e, related_ids)
        sentence_ids = get_sentence_ids_by_subevent_ids(e, related_ids)
    else:
        unit_title = ""
        unit_text = ""
        linked = []
        sentence_ids = []

    evidence_sentences = get_sentence_texts_by_ids(e, sentence_ids)
    # linked_sentences_text = join_sentence_texts(linked_sentences)

    allowed = hv_label if label_space.upper() == "L1" else L2_IDS
    actors = e["actors"]
    if actor_id == "Islamic State (IS) Militants":
        print("test")
    actor = _get_actor_by_id(actor_id, actors)
    context_sentences_wid, context_sentences = get_neighbor_context_sentences(e, sentence_ids)

    return {
        "actor": actor,
        "unit_level": unit_level,
        "unit_title": unit_title,
        "unit_text": unit_text,
        "article_title": article_title,
        "evidence_sentences": evidence_sentences,
        "context_sentences": context_sentences,
        "label_space": label_space.upper(),
        "id2label": allowed
    }

def build_input_payload_cgt(
    e: Dict[str, Any],
    iid: InstanceId,
    label_space: str,
    hv_label: str,
    hv_label_desc: str,
    related_ids: List[str]
) -> Dict[str, Any]:

    guid, unit_level, unit_id, actor_id = iid
    article_title, content = _get_article_title_content(e)

    sentence_ids = []
    # evidence based article hv recognition
    # if unit_level == "article":
    #     unit_text = ""
    #     unit_title = ""
    #     linked = []
    #     sentence_ids = get_all_subevent_sentence_ids(e)
    # raw article-based hv recognition
    if unit_level == "article":
        unit_text = content
        unit_title = article_title
        linked = []
        sentence_ids = []
    elif unit_level == "subevent":
        unit_text, sentence_ids = _find_subevent_info(e, unit_id)
        unit_title = ""
        linked = []
    elif unit_level == "behavior_chain":
        # related_ids = e["random_subevent_ids"]
        unit_title = ""
        unit_title, unit_text = _behavior_chain_unit_text_and_related_cgt(e, unit_id, related_ids)
        # linked = _get_linked_subevents(e, related_ids)
        sentence_ids = get_sentence_ids_by_subevent_ids(e, related_ids)
    elif unit_level == "story_narrative":
        unit_title, unit_text = _story_unit_text_and_related_cgt(e, unit_id, related_ids)
        # linked = _get_linked_subevents(e, related_ids, related_ids)
        sentence_ids = get_sentence_ids_by_subevent_ids(e, related_ids)
    else:
        unit_title = ""
        unit_text = ""
        linked = []
        sentence_ids = []

    evidence_sentences = get_sentence_texts_by_ids(e, sentence_ids)
    # linked_sentences_text = join_sentence_texts(linked_sentences)

    allowed = hv_label if label_space.upper() == "L1" else L2_IDS
    actors = e["actors"]
    if actor_id == "Islamic State (IS) Militants":
        print("test")
    actor = _get_actor_by_id(actor_id, actors)
    context_sentences_wid, context_sentences = get_neighbor_context_sentences(e, sentence_ids)

    return {
        "actor": actor,
        "unit_level": unit_level,
        "unit_title": unit_title,
        "unit_text": unit_text,
        "article_title": article_title,
        "evidence_sentences": evidence_sentences,
        "context_sentences": context_sentences,
        "label_space": label_space.upper(),
        "id2label": allowed
    }

def build_gold_from_hv_rows(
    hv_rows: List[Dict[str, Any]],
    label_space: str = "L1",
) -> Dict[InstanceId, Dict[str, Set[str]]]:
    """
    Returns:
      gold[iid] = {
        "aligned": set(label_ids),
        "contradictory": set(label_ids)
      }
    """
    gold: Dict[InstanceId, Dict[str, Set[str]]] = {}
    for r in hv_rows:
        guid = str(r.get("guid", "")).strip()
        unit_level = str(r.get("unit_level", "")).strip()
        unit_id = str(r.get("unit_id", "") if r.get("unit_id") is not None else "").strip()
        actor = str(r.get("actor", "")).strip()

        if not guid or not unit_level or not actor:
            continue

        direction = r.get("direction", None)  # your convention: 1 aligned, 0 contradictory
        if direction not in (0, 1, "0", "1"):
            continue
        direction = int(direction)

        if label_space.upper() == "L1":
            lab = r.get("l1_human_value", None)
            if lab is None:
                continue
            lab = str(int(lab))
        else:
            lab = r.get("l2_human_value", None)
            if lab is None:
                continue
            lab = str(int(lab))

        iid: InstanceId = (guid, unit_level, unit_id, actor)
        if iid not in gold:
            gold[iid] = {"aligned": set(), "contradictory": set()}

        if direction == 1:
            gold[iid]["aligned"].add(lab)
        else:
            gold[iid]["contradictory"].add(lab)

    return gold

# ----------------------------
# Event indexing (by guid) and text getters for 4 levels
# ----------------------------
def index_events_by_guid(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for e in events:
        guid = str(e.get("guid", "")).strip()
        if guid:
            out[guid] = e
    return out

def sample_gold_instances(
    gold: Dict[InstanceId, Dict[str, Set[str]]],
    n: int,
    seed: int = 42,
) -> Dict[InstanceId, Dict[str, Set[str]]]:
    """
    Sample N instance keys from gold dict. Each key = (guid, unit_level, unit_id, actor).
    This is the safest way to downsample because hv_rows may contain many rows per instance.
    """
    if n <= 0 or n >= len(gold):
        return gold
    rng = random.Random(seed)
    keys = list(gold.keys())
    rng.shuffle(keys)
    keep = set(keys[:n])
    return {k: gold[k] for k in keep}

def filter_events_by_gold(
    events_by_guid: Dict[str, Dict[str, Any]],
    gold: Dict[InstanceId, Dict[str, Set[str]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Keep only guid that appears in sampled gold instances. Optional (speed/memory).
    """
    keep_guids = {iid[0] for iid in gold.keys()}
    return {g: e for g, e in events_by_guid.items() if g in keep_guids}

def ensure_gold_dict(
        gold_like: Any
) -> Dict[InstanceId, Dict[str, Set[str]]]:
    """
    Support two formats:
      1) dict[InstanceId, dict] (original format)
      2) list[dict] (format loaded from JSON)
    """
    if isinstance(gold_like, dict):
        # Already in the correct format
        return gold_like

    if not isinstance(gold_like, list):
        raise TypeError(f"gold must be dict or list[dict], got {type(gold_like)}")

    # Convert list format to dict format
    gold: Dict[InstanceId, Dict[str, Set[str]]] = {}
    for r in gold_like:
        if not isinstance(r, dict):
            continue

        guid = str(r.get("guid", "")).strip()
        unit_level = str(r.get("unit_level", "")).strip()
        unit_id = str(r.get("unit_id", "") if r.get("unit_id") is not None else "").strip()
        actor = str(r.get("actor", "")).strip()

        if not guid or not unit_level or not actor:
            continue

        aligned_raw = r.get("aligned", []) or []
        contra_raw = r.get("contradictory", []) or []

        aligned = {str(x).strip() for x in aligned_raw} if isinstance(aligned_raw, list) else set()
        contradictory = {str(x).strip() for x in contra_raw} if isinstance(contra_raw, list) else set()

        iid: InstanceId = (guid, unit_level, unit_id, actor)
        gold[iid] = {"aligned": aligned, "contradictory": contradictory}

    return gold


def _get_phase(events_by_guid: Dict[str, Dict[str, Any]], guid: str) -> str:
    """Get phase_version from events"""
    e = events_by_guid.get(guid)
    if isinstance(e, dict):
        pv = str(e.get("phase_version", "")).strip()
        return pv if pv else "unknown"
    return "unknown"


def sample_gold_instances_balanced(
        gold: Any,  # support list or dict
        events_by_guid: Dict[str, Dict[str, Any]],
        n: int,
        *,
        label_universe: List[str],
        seed: int = 42,
        phase_targets: Optional[Dict[str, int]] = None,
        level_targets: Optional[Dict[str, int]] = None,
) -> Dict[InstanceId, Dict[str, Set[str]]]:
    """
    Balanced sampling with multiple constraints:
      1) Balance phase_version (v1/v2)
      2) Balance unit_level (4 types)
      3) Balance labels (54 classes; highest priority)
    """
    #  Important fix: ensure a consistent format

    if n <= 0 or n >= len(gold):
        return gold

    rng = random.Random(seed)

    # --------- Prepare data structures ----------
    iids = list(gold.keys())

    iid2labels: Dict[InstanceId, Set[str]] = {}
    iid2phase: Dict[InstanceId, str] = {}
    iid2level: Dict[InstanceId, str] = {}

    # label -> [iid...]
    label2iids: Dict[str, List[InstanceId]] = {lab: [] for lab in label_universe}

    # Global label frequencies for rare-label priority
    global_label_freq = Counter()

    for iid in iids:
        guid, unit_level, unit_id, actor = iid
        g = gold[iid]

        labs = set()
        labs |= set(g.get("aligned", set()) or [])
        labs |= set(g.get("contradictory", set()) or [])
        labs = {str(x) for x in labs if str(x) in set(label_universe)}

        if not labs:
            continue

        iid2labels[iid] = labs
        iid2phase[iid] = _get_phase(events_by_guid, guid)
        iid2level[iid] = unit_level

        for lab in labs:
            label2iids[lab].append(iid)
            global_label_freq[lab] += 1

    # Shuffle candidate lists for each label
    for lab in label_universe:
        rng.shuffle(label2iids[lab])

    # --------- Target quotas ----------
    if phase_targets is None:
        phase_targets = {"v1": n // 2, "v2": n - (n // 2)}

    if level_targets is None:
        levels = ["article", "subevent", "behavior_chain", "story_narrative"]
        base = n // 4
        rem = n - base * 4
        level_targets = {lv: base for lv in levels}
        for i in range(rem):
            level_targets[levels[i]] += 1

    # --------- Main sampling loop ----------
    selected: Set[InstanceId] = set()
    label_count = Counter({lab: 0 for lab in label_universe})
    phase_count = Counter()
    level_count = Counter()

    # Scan pointer for each label
    ptr = {lab: 0 for lab in label_universe}

    #  Important fix: record exhausted labels
    exhausted_labels: Set[str] = set()

    def _can_take(iid: InstanceId) -> bool:
        """Check phase/level constraints"""
        pv = iid2phase.get(iid, "unknown")
        lv = iid2level.get(iid, "")

        # Phase constraint, only for v1/v2
        if pv in ("v1", "v2"):
            other = "v2" if pv == "v1" else "v1"
            if phase_count[pv] >= phase_targets.get(pv, 0) and phase_count[other] < phase_targets.get(other, 0):
                return False

        # Level constraint
        if lv in level_targets:
            deficits = {k: level_targets[k] - level_count[k] for k in level_targets}
            max_need_lv = max(deficits, key=lambda k: deficits[k])
            if level_count[lv] >= level_targets[lv] and deficits.get(max_need_lv, 0) > 0:
                return False

        return True

    def _pick_neediest_label() -> Optional[str]:
        """Select the currently most needed label, excluding exhausted labels"""
        available = [lab for lab in label_universe if lab not in exhausted_labels]
        if not available:
            return None

        return min(
            available,
            key=lambda lab: (label_count[lab], global_label_freq.get(lab, 10 ** 9))
        )

    max_iterations = len(gold) * 10  #  Prevent a real infinite loop
    iteration = 0

    while len(selected) < n and iteration < max_iterations:
        iteration += 1

        lab = _pick_neediest_label()
        if lab is None:
            # All labels are exhausted; fill the remainder randomly
            remaining = [iid for iid in iid2labels.keys() if iid not in selected]
            if not remaining:
                break
            rng.shuffle(remaining)
            for iid in remaining[:min(10, n - len(selected))]:
                selected.add(iid)
                pv = iid2phase.get(iid, "unknown")
                lv = iid2level.get(iid, "")
                phase_count[pv] += 1
                level_count[lv] += 1
                for l in iid2labels[iid]:
                    label_count[l] += 1
            continue

        candidates = label2iids.get(lab, [])
        chosen = None

        # 1) First search for a candidate satisfying phase/level constraints
        i = ptr[lab]
        while i < len(candidates):
            iid = candidates[i]
            i += 1
            if iid in selected or iid not in iid2labels:
                continue
            if _can_take(iid):
                chosen = iid
                break
        ptr[lab] = i

        # 2) Relax constraints
        if chosen is None:
            i = ptr[lab]
            while i < len(candidates):
                iid = candidates[i]
                i += 1
                if iid in selected or iid not in iid2labels:
                    continue
                chosen = iid
                break
            ptr[lab] = i

        # Important fix: if this label has been fully scanned, mark it exhausted.
        if chosen is None:
            exhausted_labels.add(lab)
            continue

        # Add the selected instance
        selected.add(chosen)
        pv = iid2phase.get(chosen, "unknown")
        lv = iid2level.get(chosen, "")

        phase_count[pv] += 1
        level_count[lv] += 1

        for l in iid2labels[chosen]:
            label_count[l] += 1

    if iteration >= max_iterations:
        print(f"[WARNING] reached max iterations, sampled {len(selected)}/{n}")

    sampled = {iid: gold[iid] for iid in selected}
    return sampled


def summarize_sampled_gold(
        sampled_gold: Dict[InstanceId, Dict[str, Set[str]]],
        events_by_guid: Dict[str, Dict[str, Any]],
        label_universe: List[str],
) -> Dict[str, Any]:
    """Summarize sampling results"""
    phase_cnt = Counter()
    level_cnt = Counter()
    label_cnt = Counter({lab: 0 for lab in label_universe})

    for iid, g in sampled_gold.items():
        guid, unit_level, unit_id, actor = iid
        pv = _get_phase(events_by_guid, guid)
        phase_cnt[pv] += 1
        level_cnt[unit_level] += 1

        labs = set(g.get("aligned", set()) or []) | set(g.get("contradictory", set()) or [])
        labs = {str(x) for x in labs if str(x) in set(label_universe)}
        for l in labs:
            label_cnt[l] += 1

    total = len(sampled_gold)
    covered = sum(1 for lab in label_universe if label_cnt[lab] > 0)

    nonzero = [label_cnt[lab] for lab in label_universe if label_cnt[lab] > 0]
    min_c = min(nonzero) if nonzero else 0
    max_c = max(nonzero) if nonzero else 0
    avg_c = sum(nonzero) / len(nonzero) if nonzero else 0

    v1 = phase_cnt.get("v1", 0)
    v2 = phase_cnt.get("v2", 0)

    return {
        "n_instances": total,

        "phase_counts": dict(phase_cnt),
        "phase_ratio_v1_v2": {
            "v1": round(v1 / total, 4) if total else 0.0,
            "v2": round(v2 / total, 4) if total else 0.0,
        },

        "unit_level_counts": dict(level_cnt),
        "unit_level_ratios": {
            k: round(v / total, 4) if total else 0.0
            for k, v in level_cnt.items()
        },

        "label_coverage": {
            "covered_labels": covered,
            "total_labels": len(label_universe),
            "coverage_ratio": round(covered / len(label_universe), 4) if label_universe else 0.0,
        },
        "label_count_summary": {
            "min_nonzero": min_c,
            "max_nonzero": max_c,
            "avg_nonzero": round(avg_c, 2),
        },
        "label_counts": {lab: int(label_cnt[lab]) for lab in label_universe},
    }


def gold_to_jsonable_rows(
        gold: Dict[InstanceId, Dict[str, Set[str]]]
) -> List[Dict[str, Any]]:
    """Convert to a JSON-serializable format"""
    rows = []
    for (guid, unit_level, unit_id, actor), g in gold.items():
        rows.append({
            "guid": guid,
            "unit_level": unit_level,
            "unit_id": unit_id,
            "actor": actor,
            "aligned": sorted(list(g.get("aligned", set()))),
            "contradictory": sorted(list(g.get("contradictory", set()))),
        })
    return rows

def list_to_cgt_gold_dict(data_list: List[Dict[str, Any]]) -> Dict[InstanceId, Dict[str, Set[str]]]:
    """
    Convert list[dict] format to dict[InstanceId, dict] format

    Input format:
    [
        {
            'guid': 'xxx',
            'unit_level': 'article',
            'unit_id': '',
            'actor': 'yyy',
            'aligned': ['11', '25'],
            'contradictory': ['27']
        },
        ...
    ]

    Output format:
    {
        ('guid', 'unit_level', 'unit_id', 'actor'): {
            'aligned': {'11', '25'},
            'contradictory': {'27'}
        },
        ...
    }
    """
    gold: Dict[InstanceId, Dict[str, Set[str]]] = {}

    for item in data_list:
        if not isinstance(item, dict):
            continue

        # Extract the four key fields
        guid = str(item.get("guid", "")).strip()
        unit_level = str(item.get("unit_level", "")).strip()
        unit_id = str(item.get("unit_id", "") if item.get("unit_id") is not None else "").strip()
        actor = str(item.get("actor", "")).strip()
        random_subevent_ids = str(item.get("random_subevent_ids", "") if item.get("random_subevent_ids") is not None else "").strip()

        # Check required fields
        if not guid or not unit_level or not actor:
            print(f"[WARNING] Skipping item with missing fields: guid={guid}, unit_level={unit_level}, actor={actor}")
            continue

        # Extract aligned and contradictory labels
        aligned_raw = item.get("aligned", [])
        contradictory_raw = item.get("contradictory", [])

        # Convert to set[str]
        aligned = set()
        if isinstance(aligned_raw, list):
            aligned = {str(x).strip() for x in aligned_raw if x is not None and str(x).strip()}
        elif aligned_raw is not None:
            aligned = {str(aligned_raw).strip()}

        contradictory = set()
        if isinstance(contradictory_raw, list):
            contradictory = {str(x).strip() for x in contradictory_raw if x is not None and str(x).strip()}
        elif contradictory_raw is not None:
            contradictory = {str(contradictory_raw).strip()}

        # Build the key
        iid: InstanceId = (guid, unit_level, unit_id, actor)

        # If the same iid appears multiple times, merge labels; this should not happen in theory
        if iid in gold:
            print(f"[WARNING] Duplicate iid found: {iid}, merging labels")
            gold[iid]["aligned"] |= aligned
            gold[iid]["contradictory"] |= contradictory
        else:
            gold[iid] = {
                "aligned": aligned,
                "contradictory": contradictory,
                "random_subevent_ids": random_subevent_ids
            }

    return gold

def list_to_gold_dict(data_list: List[Dict[str, Any]]) -> Dict[InstanceId, Dict[str, Set[str]]]:
    """
    Convert list[dict] format to dict[InstanceId, dict] format

    Input format:
    [
        {
            'guid': 'xxx',
            'unit_level': 'article',
            'unit_id': '',
            'actor': 'yyy',
            'aligned': ['11', '25'],
            'contradictory': ['27']
        },
        ...
    ]

    Output format:
    {
        ('guid', 'unit_level', 'unit_id', 'actor'): {
            'aligned': {'11', '25'},
            'contradictory': {'27'}
        },
        ...
    }
    """
    gold: Dict[InstanceId, Dict[str, Set[str]]] = {}

    for item in data_list:
        if not isinstance(item, dict):
            continue

        # Extract the four key fields
        guid = str(item.get("guid", "")).strip()
        unit_level = str(item.get("unit_level", "")).strip()
        unit_id = str(item.get("unit_id", "") if item.get("unit_id") is not None else "").strip()
        actor = str(item.get("actor", "")).strip()

        # Check required fields
        if not guid or not unit_level or not actor:
            print(f"[WARNING] Skipping item with missing fields: guid={guid}, unit_level={unit_level}, actor={actor}")
            continue

        # Extract aligned and contradictory labels
        aligned_raw = item.get("aligned", [])
        contradictory_raw = item.get("contradictory", [])

        # Convert to set[str]
        aligned = set()
        if isinstance(aligned_raw, list):
            aligned = {str(x).strip() for x in aligned_raw if x is not None and str(x).strip()}
        elif aligned_raw is not None:
            aligned = {str(aligned_raw).strip()}

        contradictory = set()
        if isinstance(contradictory_raw, list):
            contradictory = {str(x).strip() for x in contradictory_raw if x is not None and str(x).strip()}
        elif contradictory_raw is not None:
            contradictory = {str(contradictory_raw).strip()}

        # Build the key
        iid: InstanceId = (guid, unit_level, unit_id, actor)

        # If the same iid appears multiple times, merge labels; this should not happen in theory
        if iid in gold:
            print(f"[WARNING] Duplicate iid found: {iid}, merging labels")
            gold[iid]["aligned"] |= aligned
            gold[iid]["contradictory"] |= contradictory
        else:
            gold[iid] = {
                "aligned": aligned,
                "contradictory": contradictory
            }

    return gold

def process_and_store_bak(filename, data_list, client):
    time.sleep(0.5)
    processed_count = 0
    total_count = len(data_list)
    print(f"Process ID: {os.getpid()}, Thread ID: {threading.get_ident()}, filename: {filename}\n")
    for idx, prompt_con in enumerate(data_list):
        local_guid = prompt_con["guid"]
        unit_level = prompt_con["unit_level"]
        unit_id = prompt_con["unit_id"]
        actor = prompt_con["actor"]
        role1 = prompt_con["role1"]
        content1 = prompt_con["prompt_system"]
        role2 = prompt_con["role2"]
        api_model = prompt_con["model_name"]
        content2 = prompt_con["user_prompt"]
        output_file_root_path = prompt_con["output_file_root_path"]
        output_completed_file_root_path = prompt_con["output_completed_file_root_path"]
        print("data_idx: " + str(idx) + "\n")
        try:
            start_time = time.time()
            if content2:
                # completion = llm_utils.call_open_api(api_model, client, role1, content1, role2, content2)
                if api_model.startswith("gemini"):
                    completion = llm_utils.call_google_gemini(api_model, client, role1, content1, role2, content2)
                elif api_model.startswith("claude"):
                    completion = llm_utils.call_anthropic_stream(api_model, client, role1, content1, role2, content2)
                elif "o3-deep-research" in api_model:
                    completion = llm_utils.call_open_api_deepresearch(api_model, client, role1, content1, role2, content2)
                else:
                    completion = llm_utils.call_open_api(api_model, client, role1, content1, role2, content2)
            end_time = time.time()
            hours, minutes, seconds = preprocess_utils.spent_time(start_time, end_time)

            # ************TEST************
            # args = parser.parse_args()
            log_con = "*******current file name: " + str(filename) + ",total count:" + str(total_count) + ", api_model:" + api_model + ", current num: " + str(idx) + f", SpentTimeforPerArticle: {hours}h {minutes}m {seconds:.2f}s" + "********"
            print(log_con)
            # ************TEST************
            # file_index = preprocess_utils.compute_subfile_index(idx, process_start, max_rows_per_file)
            # input_count = token_utils.num_tokens_from_string(str(content2), "cl100k_base")
            if completion:
                print("********Input token count: " + str(completion.usage.prompt_tokens))
                print("********response token count: " + str(completion.usage.completion_tokens))

            output_file = output_file_root_path + "/" + str(filename) + "/"
            output_completed_file = output_completed_file_root_path + "/" + str(filename) + "/"

            file_utils.check_and_create_file(output_file)
            file_utils.check_and_create_file(output_completed_file)

            output_file_name = output_file + "output_results.json"
            output_completed_file_name = output_completed_file + "completed_results.json"

            if content2:
                if api_model.startswith("gemini"):
                    comp_message = completion.candidates[0].content
                    comp_content = completion.text
                elif api_model.startswith("claude"):
                    comp_message = completion
                    comp_content = completion
                elif "o3-deep-research" in api_model:
                    comp_message = completion.output[-1].content[0].text
                    comp_content = completion.output[-1].content[0].text
                else:
                    comp_message = completion.choices[0].message
                    comp_content = completion.choices[0].message.content
            else:
                comp_message = ""
                comp_content = "{}"

            if api_model.startswith("claude"):
                # Clean whitespace
                result_con = comp_content.strip()
            else:
                match = re.search(r"```json\s*(.*?)\s*```", comp_content, re.DOTALL)
                if match:
                    result_con = match.group(1)
                else:
                    result_con = comp_content
                    print("No JSON section found")

            # current news
            json_object = json.loads(result_con)
            json_object["guid"] = local_guid
            json_object["unit_level"] = unit_level
            json_object["unit_id"] = unit_id
            json_object["actor"] = actor
            file_utils.append_json_data_to_file(output_file_name, json_object, 4)

            # completed_guid = "{\"guid\": \"" + local_guid.replace("\n", "\\n") + "\"}"
            completed_instance = (
                    "{\"guid\": \"" + str(local_guid).replace("\n", "\\n") + "\""
                    + ", \"unit_level\": \"" + str(unit_level).replace("\n", "\\n") + "\""
                    + ", \"unit_id\": \"" + str(unit_id).replace("\n", "\\n") + "\""
                    + ", \"actor\": \"" + str(actor).replace("\n", "\\n") + "\"}"
            )

            json_object_completed_guid = json.loads(str(completed_instance))
            file_utils.append_json_data_to_file(output_completed_file_name, json_object_completed_guid, 4)

            processed_count += 1
            print("guid:" + local_guid + " is processed!")
        except Exception as e:
            print(f"Error result: {e}")
            continue

def process_and_store(filename, data_list, client):
    time.sleep(0.5)
    processed_count = 0
    total_count = len(data_list)
    print(f"Process ID: {os.getpid()}, Thread ID: {threading.get_ident()}, filename: {filename}\n")

    for idx, prompt_con in enumerate(data_list):
        local_guid = prompt_con["guid"]
        unit_level = prompt_con["unit_level"]
        unit_id = prompt_con["unit_id"]
        actor = prompt_con["actor"]

        role1 = prompt_con["role1"]
        content1 = prompt_con["prompt_system"]
        role2 = prompt_con["role2"]

        api_model = prompt_con["model_name"]
        content2 = prompt_con["user_prompt"]

        #  Optional MapReduce parameters, default False and backward-compatible
        use_mr = bool(prompt_con.get("use_map_reduce", False))
        payload = prompt_con.get("payload")
        prompt_variant = prompt_con.get("prompt_variant")
        prompt_tpl = prompt_con.get("prompt_content_template")
        prompt_system = prompt_con.get("prompt_system")
        max_len = int(prompt_con.get("max_len", 4096))
        mr_window = int(prompt_con.get("mr_window", 1400))
        mr_overlap = int(prompt_con.get("mr_overlap", 200))
        group = prompt_con.get("group")
        model_name = prompt_con.get("model_name")

        output_file_root_path = prompt_con["output_file_root_path"]
        output_completed_file_root_path = prompt_con["output_completed_file_root_path"]

        print("data_idx: " + str(idx) + "\n")

        try:
            start_time = time.time()

            final_obj = None
            completion = None  #  Prevent completion from being undefined in the MR branch

            # ====== MapReduce branch: triggered only when use_mr, payload/template exists, and input is too long ======
            if content2 and use_mr and isinstance(payload, dict) and isinstance(prompt_tpl, str) and isinstance(prompt_system, str):
                prompt_tokens = _count_tokens_api(content2, api_model)
                overflow = prompt_tokens > max_len

                if overflow:
                    unit_text = payload.get("unit_text") or ""
                    evidence = payload.get("evidence_sentences") or []

                    unit_len = _count_tokens_api(unit_text, api_model)
                    ev_len = 0
                    if isinstance(evidence, list):
                        ev_len = sum(
                            _count_tokens_api((x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)), api_model)
                            for x in evidence
                        )

                    RATIO_THRESHOLD = 1.5
                    chunk_payloads = []

                    if unit_len > ev_len * RATIO_THRESHOLD and isinstance(unit_text, str) and unit_text.strip():
                        pieces = _chunk_text_by_tokens_api(unit_text, api_model, window=mr_window, overlap=mr_overlap)
                        for piece in pieces:
                            p2 = json.loads(json.dumps(payload, ensure_ascii=False))
                            p2["unit_text"] = piece
                            chunk_payloads.append(p2)
                    else:
                        ev_chunks = _chunk_evidence_sentences_by_tokens_api(
                            evidence if isinstance(evidence, list) else [],
                            api_model,
                            window=mr_window,
                            overlap=mr_overlap
                        )
                        for ev in ev_chunks:
                            p2 = json.loads(json.dumps(payload, ensure_ascii=False))
                            p2["evidence_sentences"] = ev
                            chunk_payloads.append(p2)

                    chunk_objs = []
                    for pch in chunk_payloads:
                        content2_ch = prompt_utils.build_prompt_text(pch, prompt_variant, prompt_tpl, prompt_system, model_name, group)

                        if api_model.startswith("gemini"):
                            comp = llm_utils.call_google_gemini(api_model, client, role1, content1, role2, content2_ch)
                            comp_content = comp.text
                        elif api_model.startswith("claude"):
                            comp = llm_utils.call_anthropic_stream(api_model, client, role1, content1, role2, content2_ch)
                            comp_content = comp
                        elif "o3-deep-research" in api_model:
                            comp = llm_utils.call_open_api_deepresearch(api_model, client, role1, content1, role2, content2_ch)
                            comp_content = comp.output[-1].content[0].text
                        else:
                            comp = llm_utils.call_open_api(api_model, client, role1, content1, role2, content2_ch)
                            comp_content = comp.choices[0].message.content

                        obj = _extract_json_from_completion_text(api_model, comp_content)
                        chunk_objs.append(obj)

                    final_obj = _reduce_union_prefer_majority(chunk_objs)

            # ====== Legacy branch: use the original single call when MR is not triggered ======
            if content2 and final_obj is None:
                if api_model.startswith("gemini"):
                    completion = llm_utils.call_google_gemini(api_model, client, role1, content1, role2, content2)
                    comp_content = completion.text
                elif api_model.startswith("claude"):
                    completion = llm_utils.call_anthropic_stream(api_model, client, role1, content1, role2, content2)
                    comp_content = completion
                elif "o3-deep-research" in api_model:
                    completion = llm_utils.call_open_api_deepresearch(api_model, client, role1, content1, role2, content2)
                    comp_content = completion.output[-1].content[0].text
                else:
                    completion = llm_utils.call_open_api(api_model, client, role1, content1, role2, content2)
                    comp_content = completion.choices[0].message.content

                final_obj = _extract_json_from_completion_text(api_model, comp_content)

            end_time = time.time()
            hours, minutes, seconds = preprocess_utils.spent_time(start_time, end_time)

            log_con = (
                "*******current file name: " + str(filename)
                + ",total count:" + str(total_count)
                + ", api_model:" + api_model
                + ", current num: " + str(idx)
                + f", SpentTimeforPerArticle: {hours}h {minutes}m {seconds:.2f}s"
                + "********"
            )
            print(log_con)

            #  completion exists only in the legacy single-call branch
            if completion is not None and hasattr(completion, "usage"):
                try:
                    print("********Input token count: " + str(completion.usage.prompt_tokens))
                    print("********response token count: " + str(completion.usage.completion_tokens))
                except Exception:
                    pass

            output_file = output_file_root_path + "/" + str(filename) + "/"
            output_completed_file = output_completed_file_root_path + "/" + str(filename) + "/"

            file_utils.check_and_create_file(output_file)
            file_utils.check_and_create_file(output_completed_file)

            output_file_name = output_file + "output_results.json"
            output_completed_file_name = output_completed_file + "completed_results.json"

            #  final_obj is the unified output for MR or single-call paths
            json_object = final_obj if isinstance(final_obj, dict) else {
                "aligned_with_human_values": [],
                "contradictory_to_human_values": []
            }

            json_object["guid"] = local_guid
            json_object["unit_level"] = unit_level
            json_object["unit_id"] = unit_id
            json_object["actor"] = actor

            file_utils.append_json_data_to_file(output_file_name, json_object, 4)

            #  Preserve the original completed_results.json behavior
            completed_instance = (
                "{\"guid\": \"" + str(local_guid).replace("\n", "\\n") + "\""
                + ", \"unit_level\": \"" + str(unit_level).replace("\n", "\\n") + "\""
                + ", \"unit_id\": \"" + str(unit_id).replace("\n", "\\n") + "\""
                + ", \"actor\": \"" + str(actor).replace("\n", "\\n") + "\"}"
            )
            json_object_completed_guid = json.loads(str(completed_instance))
            file_utils.append_json_data_to_file(output_completed_file_name, json_object_completed_guid, 4)

            processed_count += 1
            print("guid:" + str(local_guid) + " is processed!")

        except Exception as e:
            print(f"Error result: {e}")
            continue

def thread_processing(prompt_cons_file, min_thread_count, client):
    tasks = list(prompt_cons_file.items())
    max_workers = min(min_thread_count, cpu_count())
    print(f"CPU cores: {cpu_count()}, threads: {max_workers}")

    # Run tasks with multiple threads
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_task = {
            executor.submit(process_and_store, filename, data_list, client): (filename, data_list, client)
            for filename, data_list in tasks
        }

        # Wait for tasks to finish and collect results
        for future in as_completed(future_to_task):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print("Error while executing task:", e)

def split_test_data(
    gold: Dict[InstanceId, Dict[str, Set[str]]],
    group_size: int = 100,
) -> List[Dict[InstanceId, Dict[str, Set[str]]]]:
    """
    Split gold dict into multiple groups, each containing up to `group_size` instances.
    - Keep original insertion order (no shuffling).
    - Each group is a dict with the same {InstanceId: label_dict} structure.

    Args:
        gold: dict of InstanceId -> {"aligned": set(...), "contradictory": set(...)} (or similar)
        group_size: number of instances per group (default 100)

    Returns:
        List of gold-sub-dicts.
    """
    if not isinstance(gold, dict) or not gold:
        return []

    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    items = list(gold.items())  # preserves insertion order in Python 3.7+
    out: List[Dict[InstanceId, Dict[str, Set[str]]]] = []

    for i in range(0, len(items), group_size):
        chunk_items = items[i : i + group_size]
        out.append(dict(chunk_items))

    return out

def list_to_dict(
    rows: List[Dict[str, Any]],
    key_fields: Tuple[str, str, str, str] = ("guid", "unit_level", "unit_id", "actor"),
) -> Dict[InstanceId, bool]:
    """
    Convert:
      [{"guid":..., "unit_id":..., "unit_level":..., "actor":...}, ...]
    into:
      { (guid, unit_id, unit_level, actor): True, ... }
    """
    out: Dict[InstanceId, bool] = {}
    for i, r in enumerate(rows):
        missing = [f for f in key_fields if f not in r]
        if missing:
            raise KeyError(f"Row {i} missing fields: {missing}. Row={r}")

        key: InstanceId = tuple(str(r[f]) for f in key_fields)  # type: ignore
        out[key] = True
    return out


# -------- Improved chunking decision logic --------

# Calculate token lengths for each content type
def calc_token_len(tokenizer, text: str) -> int:
    """Calculate token length of text"""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def calc_sentences_token_len(tokenizer, sentences: List[Dict[str, Any]]) -> int:
    """Calculate total token length of sentence list"""
    if not isinstance(sentences, list) or not sentences:
        return 0
    total = 0
    for sent in sentences:
        if isinstance(sent, dict):
            text = sent.get("sentence") or sent.get("text") or ""
            if isinstance(text, str):
                total += calc_token_len(tokenizer, text)
    return total

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def normalize_label_list(x: Any) -> List[str]:
    """
    Input may be list, None, or another type; normalize it to a list of label-id strings.
    Remove empty values and duplicates; sorting is deferred because sets are used later.
    """
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for it in x:
            s = _as_str(it).strip()
            if s != "":
                out.append(s)
        return out
    # Some abnormal outputs may be a single string or number
    s = _as_str(x).strip()
    return [s] if s else []

def normalize_pred_labels(obj: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """
    Support these output fields:
      aligned_with_human_values / contradictory_to_human_values
    Ensure aligned and contradictory do not overlap; by default keep aligned and drop contradictory on conflicts.
    """
    a_list = normalize_label_list(obj.get("aligned_with_human_values"))
    c_list = normalize_label_list(obj.get("contradictory_to_human_values"))

    a_set = set(a_list)
    c_set = set(c_list)

    overlap = a_set & c_set
    if overlap:
        # Conflict policy: keep aligned and remove duplicates from contradictory
        c_set -= overlap
    return a_set, c_set

def build_pred_from_records(records: List[Dict[str, Any]]) -> Dict[InstanceId, Dict[str, Set[str]]]:
    pred: Dict[InstanceId, Dict[str, Set[str]]] = {}
    dup_counter = Counter()
    conflict_counter = Counter()

    for r in records:
        guid = _as_str(r.get("guid")).strip()
        unit_level = _as_str(r.get("unit_level")).strip()
        unit_id = _as_str(r.get("unit_id")).strip()  # article may be ""
        actor = _as_str(r.get("actor")).strip()

        if not guid or not unit_level or not actor:
            # Skip records that cannot form a key; this can be changed to raise
            continue

        iid: InstanceId = (guid, unit_level, unit_id, actor)

        # Compatibility: records may already contain {aligned_with_human_values:[], contradictory_to...:[]}
        a_set, c_set = normalize_pred_labels(r)

        # If the same iid appears in multiple files, merge with union; this should not happen in theory
        if iid in pred:
            dup_counter["dup_iid"] += 1
            before_overlap = (pred[iid]["aligned"] & pred[iid]["contradictory"])
            pred[iid]["aligned"] |= a_set
            pred[iid]["contradictory"] |= c_set
            after_overlap = (pred[iid]["aligned"] & pred[iid]["contradictory"])
            if after_overlap:
                # Run conflict cleanup again: keep aligned
                pred[iid]["contradictory"] -= after_overlap
                conflict_counter["overlap_fixed_after_union"] += 1
        else:
            pred[iid] = {"aligned": set(a_set), "contradictory": set(c_set)}

        # A single record may also overlap; normalize has already removed contradictory duplicates
        # Only collect statistics here
        if set(normalize_label_list(r.get("aligned_with_human_values"))) & set(normalize_label_list(r.get("contradictory_to_human_values"))):
            conflict_counter["overlap_in_single_record"] += 1

    if dup_counter:
        print(f"[WARN] Found duplicated InstanceId entries across files: {dup_counter}")
    if conflict_counter:
        print(f"[WARN] Found aligned/contradictory overlap situations (auto-fixed): {conflict_counter}")

    return pred

def pred_to_serializable(pred: Dict[InstanceId, Dict[str, Set[str]]]) -> List[Dict[str, Any]]:
    serializable = []
    for iid, d in pred.items():
        guid, unit_level, unit_id, actor = iid

        # Sort numerically when possible; otherwise sort as strings
        def sort_key(x: str):
            return (0, int(x)) if x.isdigit() else (1, x)

        serializable.append({
            "guid": guid,
            "unit_level": unit_level,
            "unit_id": unit_id,
            "actor": actor,
            "aligned_with_human_values": sorted(list(d["aligned"]), key=sort_key),
            "contradictory_to_human_values": sorted(list(d["contradictory"]), key=sort_key),
        })
    return serializable

def count_empty_pred_instances(pred: Dict[InstanceId, Dict[str, Set[str]]]) -> int:
    """
    Count instances in pred where both aligned and contradictory are empty
    """
    empty = 0
    for iid, d in pred.items():
        aligned = d.get("aligned", set()) or set()
        contra = d.get("contradictory", set()) or set()
        if len(aligned) == 0 and len(contra) == 0:
            empty += 1
    return empty

def _get_api_encoding(model_name: str):
    """
    Best-effort tokenizer for API models.
    - Prefer tiktoken if available
    - Fallback to naive char-based estimate
    """
    if tiktoken is None:
        return None
    # Rule of thumb: cl100k_base is often sufficient for chunk control in OpenAI-compatible models
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def _count_tokens_api(text: str, model_name: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    enc = _get_api_encoding(model_name)
    if enc is None:
        # Rough estimate: English is about 4 chars/token; Chinese/Japanese is more complex but sufficient for triggering MR
        return max(1, len(text) // 4)
    return len(enc.encode(text))

def _chunk_text_by_tokens_api(text: str, model_name: str, window: int, overlap: int) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return [""]
    enc = _get_api_encoding(model_name)
    if enc is None:
        # fallback: split by characters approximately
        step = max(1, window - overlap)
        out = []
        s = 0
        while s < len(text):
            e = min(len(text), s + window * 4)
            out.append(text[s:e])
            if e >= len(text):
                break
            s = max(0, e - overlap * 4)
        return out

    ids = enc.encode(text)
    if len(ids) <= window:
        return [text]
    step = max(1, window - overlap)
    chunks = []
    for s in range(0, len(ids), step):
        e = min(len(ids), s + window)
        chunks.append(enc.decode(ids[s:e]))
        if e >= len(ids):
            break
    return chunks

def _chunk_evidence_sentences_by_tokens_api(evidence_sentences: list, model_name: str, window: int, overlap: int) -> list[list[dict]]:
    """
    evidence_sentences: list[str] or list[dict] (evidence_sentences is currently list[str] in build_input_payload)
    Return list[chunks]; each chunk is list[dict|str] and preserves the original structure
    """
    if not isinstance(evidence_sentences, list) or not evidence_sentences:
        return [[]]

    # Normalize text extraction
    def get_text(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("sentence") or x.get("text") or ""
        return ""

    lens = [_count_tokens_api(get_text(x), model_name) for x in evidence_sentences]

    chunks = []
    n = len(evidence_sentences)
    start = 0
    while start < n:
        total = 0
        end = start
        while end < n and total + lens[end] <= window:
            total += lens[end]
            end += 1
        if end == start:
            end = start + 1

        chunks.append(evidence_sentences[start:end])
        if end >= n:
            break

        # token overlap
        back = 0
        new_start = end
        while new_start > start and back < overlap:
            new_start -= 1
            back += lens[new_start]
        start = new_start

    return chunks

def _extract_json_from_completion_text(api_model: str, comp_content: str) -> dict:
    """
    Reuse the current logic: prefer ```json ...```, otherwise call json.loads directly.
    Return an empty structure on errors.
    """
    if not isinstance(comp_content, str):
        return {"aligned_with_human_values": [], "contradictory_to_human_values": []}

    if api_model.startswith("claude"):
        s = comp_content.strip()
    else:
        m = re.search(r"```json\s*(.*?)\s*```", comp_content, re.DOTALL)
        s = m.group(1) if m else comp_content

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            obj.setdefault("aligned_with_human_values", [])
            obj.setdefault("contradictory_to_human_values", [])
            return obj
    except Exception:
        pass

    return {"aligned_with_human_values": [], "contradictory_to_human_values": []}

def _reduce_union_prefer_majority(chunk_objs: list[dict]) -> dict:
    """
    union plus deduplication plus prefer_majority conflict resolution; drop contradictory on ties
    """
    a_cnt = Counter()
    c_cnt = Counter()

    for obj in chunk_objs:
        a = obj.get("aligned_with_human_values", []) or []
        c = obj.get("contradictory_to_human_values", []) or []
        if not isinstance(a, list): a = []
        if not isinstance(c, list): c = []
        for x in a:
            xs = str(x).strip()
            if xs:
                a_cnt[xs] += 1
        for x in c:
            xs = str(x).strip()
            if xs:
                c_cnt[xs] += 1

    a_all = set(a_cnt.keys())
    c_all = set(c_cnt.keys())

    conflicts = a_all & c_all
    for k in list(conflicts):
        if a_cnt[k] > c_cnt[k]:
            c_all.discard(k)
        elif c_cnt[k] > a_cnt[k]:
            a_all.discard(k)
        else:
            # On ties, drop contradictory to avoid dual polarity for the same label
            c_all.discard(k)

    def _sort(xs):
        try:
            return sorted(xs, key=lambda z: int(z))
        except Exception:
            return sorted(xs)

    return {
        "aligned_with_human_values": _sort(a_all),
        "contradictory_to_human_values": _sort(c_all),
    }
