"""Prompt assembly helpers for human-value recognition experiments."""

import os
import json
import sys
import time
from dataset.wiki_ontology.new_ontology_hv import tr_ag
from dataset.wiki_ontology.event_genres import genre
from dataset.behavior.behavior_relations import behavior_relation_types
from utils import data_utils
from typing import Any, Dict, List, Tuple, Set, Optional, Union

def get_hv_values(hv_file_path):
    with open(hv_file_path, 'r') as file:
        hv_values = json.load(file)
    hv_names = [hv_value["level-1"] for hv_value in hv_values["values"]]
    return str(hv_names)

def get_hv1_mappings_str(hv_file_path):
    with open(hv_file_path, 'r') as file:
        hv_values = json.load(file)
    return str(hv_values)

def get_hv1_mappings(hv_file_path):
    with open(hv_file_path, 'r') as file:
        hv_values = json.load(file)
    return hv_values

def get_hv_description(hv_file_path):
    with open(hv_file_path, 'r') as file:
        data = json.load(file)
    # Generate new dictionaries using iteration and copies
    result = {}
    for entry in data["values"]:
        key = entry["level-1"]
        # Copy the dictionary and remove the name key
        entry_copy = entry.copy()
        del entry_copy["level-1"]
        result[key] = entry_copy
    return str(result)

def get_hv_description_dict(hv_file_path):
    with open(hv_file_path, 'r') as file:
        data = json.load(file)
    # Generate new dictionaries using iteration and copies
    result = {}
    for entry in data["values"]:
        key = entry["level-1"]
        # Copy the dictionary and remove the name key
        entry_copy = entry.copy()
        del entry_copy["level-1"]
        result[key] = entry_copy
    return result

def find_subevent_id(subevent_human_values):
    for subeventid, subevent_value in subevent_human_values.items():
        if subeventid == "subevent_id":
            return subevent_value
    return None

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

def find_subevents(related_subevent_ids, subevents):
    related_subevents = []
    for subevent in subevents:
        if subevent["id"] in related_subevent_ids:
            related_subevents.append(subevent["subevent"])
    return related_subevents

def find_subevents_with_id(related_subevent_ids, subevents):
    related_subevents = []
    for subevent in subevents:
        new_subevent = {}
        if subevent["id"] in related_subevent_ids:
            new_subevent["subevent_id"] = subevent["id"]
            new_subevent["subevent"] = subevent["subevent"]
            related_subevents.append(new_subevent)
    return related_subevents

def find_hv_specifics(specifc_type, search_key, specific_human_values):
    for specific_item in specific_human_values:
        if search_key == specific_item[specifc_type]:
            return specific_item
    return None

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


def find_behavior_chains(behavior_ids_chains, behavior_chains):
    cross_behavior_chains = []
    for behavior_chain in behavior_chains:
        temp_behavior_chain_ids = []
        cross_chains = {}
        if isinstance(behavior_chain, str):
            print("behavior chain is str obj")
        for behavior_chain_title, behaviors in behavior_chain.items():
            cross_behavior = []
            for behavior in behaviors:
                behavior_id = behavior["id"]
                if behavior_id in behavior_ids_chains:
                    cross_behavior.append(behavior)
                temp_behavior_chain_ids.append(behavior_id)
            if cross_behavior:
                cross_chains[behavior_chain_title] = cross_behavior
        if cross_chains:
            cross_behavior_chains.append(cross_chains)
        if behavior_ids_chains == temp_behavior_chain_ids:
            return behavior_chain, "behavior_chain"
    return cross_behavior_chains, "cross_behavior_chains"

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

def check_processed(actor, hv_label, processed_directional_assessment_results):
    for processed_directional_assessment_result in processed_directional_assessment_results:
        processed_actor = processed_directional_assessment_result["actor"]
        processed_human_value_label = processed_directional_assessment_result["human_value_label"]
        if actor == processed_actor and hv_label == processed_human_value_label:
            return True
    return False

def find_unprocessed_hvs(source_directional_human_values, processed_directional_assessment_results):
    unprocessed_human_values = {}
    for actor, human_value_lists in source_directional_human_values.items():
        unprocessed_human_value_lists = []
        for hv_detail in human_value_lists:
            unprocessed_hv_detail = {}
            for hv_label, hv_desc in hv_detail.items():
                check_results = check_processed(actor, hv_label, processed_directional_assessment_results)
                if not check_results:
                    unprocessed_hv_detail.update(hv_detail)
            if unprocessed_hv_detail:
                unprocessed_human_value_lists.append(unprocessed_hv_detail)
        if unprocessed_human_value_lists:
            unprocessed_human_values[actor] = unprocessed_human_value_lists
    return unprocessed_human_values

def build_input_json(
    *,
    actor: str,
    unit_level: str,
    unit_text: str,
    article_title: str,
    article_content: str,
    linked_subevents: Optional[Union[List[str], List[Dict[str, Any]]]] = None,
    allowed_label_ids: Optional[List[Union[int, str]]] = None,
    label_definitions: Optional[Dict[Union[int, str], str]] = None,
) -> str:
    """
    Build a STRICT, valid JSON string for your prompt input.
    - Safely escapes newlines/quotes/backslashes in article_content and unit_text automatically.
    - Returns a JSON string (optionally pretty-printed by setting indent).

    NOTE:
    - Do NOT manually replace into a template. Construct dict -> json.dumps instead.
    """
    data: Dict[str, Any] = {
        "actor": actor,
        "unit_level": unit_level,  # e.g., "article" | "subevent" | "narrative_composite" | "behavior_composite"
        "unit_text": unit_text,
        "article_context": {
            "title": article_title,
            "content": article_content,
        },
        "linked_subevents": linked_subevents if linked_subevents is not None else [],
        "allowed_label_ids": allowed_label_ids if allowed_label_ids is not None else [],
    }

    # Optional: provide label definitions if you want (helps zero-shot / cross-model)
    if label_definitions is not None:
        # JSON object keys must be strings -> convert keys to str
        data["label_definitions"] = {str(k): v for k, v in label_definitions.items()}

    # ensure_ascii=False keeps Japanese/Chinese readable; json.dumps handles escaping (\n, ", \, etc.)
    return json.dumps(data, ensure_ascii=False)

def make_input_obj(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the <INPUT_VALUE> object (Python dict), customized by unit_level.
    Do NOT json.dumps fields here; only dump the final object.
    """
    actor = payload["actor"]
    unit_level = payload["unit_level"]
    unit_level = LEVEL_MAP.get(unit_level, unit_level)

    # common fields
    input_obj: Dict[str, Any] = {
        "actor": actor,
        "unit_level": unit_level,   # optional    # summary for this unit
        "article_title": payload.get("article_title", ""),
        "id2label": payload.get("id2label", {}),         # { "0": "Be creative", ... } or {0: "..."} both ok
    }

    # evidence sentences: list[{"sentence_id": "...", "sentence": "..."}]
    evidence_sentences = payload.get("evidence_sentences", [])
    context_sentences  = payload.get("context_sentences", [])

    # normalize types to avoid crashes
    if not isinstance(evidence_sentences, list):
        evidence_sentences = []
    if not isinstance(context_sentences, list):
        context_sentences = []

    # ---- per-level customization ----
    if unit_level == "subevent":
        # subevent: unit_text is the subevent summary; evidence_sentences are the source sentences
        input_obj["unit_text"]  = payload.get("unit_text", "")   # prev/next for disambiguation
        input_obj["evidence_sentences"] = evidence_sentences
        input_obj["context_sentences"]  = context_sentences   # prev/next for disambiguation

    elif unit_level in ("behavior_composite_event", "story_composite_event"):
        # behavior/story: keep summary + evidence; optionally include subevents_brief if you have it
        input_obj["unit_text"]  = payload.get("unit_text", "")
        input_obj["evidence_sentences"] = evidence_sentences
        input_obj["context_sentences"]  = context_sentences

    elif unit_level == "article":
        # article: evidence can be all related sentences; context can be 1-2 theme sentences
        # evidence-based article-level human value recognition
        # input_obj["unit_text"]  = ""   # prev/next for disambiguation
        # input_obj["evidence_sentences"] = evidence_sentences
        # input_obj["context_sentences"]  = context_sentences

        # raw article-level human value recogntion
        input_obj["unit_text"]  = payload.get("unit_text", "")   # prev/next for disambiguation
        input_obj["evidence_sentences"] = evidence_sentences
        input_obj["context_sentences"]  = context_sentences

    else:
        # fallback: still pass what you have
        input_obj["unit_text"]  = ""   # prev/next for disambiguation
        input_obj["evidence_sentences"] = evidence_sentences
        input_obj["context_sentences"]  = context_sentences

    return input_obj

def build_input_json_pretty(**kwargs) -> str:
    """Pretty-printed variant (still valid JSON)."""
    data = json.loads(build_input_json(**kwargs))
    return json.dumps(data, ensure_ascii=False)
    # return json.dumps(data, ensure_ascii=False, indent=2)

import json
from typing import Any, Dict, List

LEVEL_MAP = {
    "story_narrative": "story_composite_event",
    "behavior_chain": "behavior_composite_event",
}

def build_prompt_text(payload: Dict[str, Any], variant: str, prompt_content: str, prompt_system: str, model_name: str, group: str,
    pretty: bool = False) -> str:
    """
    variant:
      A: vanilla instruction-only (canonical)
      B/C/D: you can extend later; for training I strongly recommend keep A as canonical.
    """
    input_obj = make_input_obj(payload)
    json_str = json.dumps(
        input_obj,
        ensure_ascii=False,
        indent=2 if pretty else None
    )

    prompt_content = prompt_content.replace("<INPUT_VALUE>", json_str)
    if group != "G1":
        if "Qwen" in model_name:
            prompt = f"/no_think\n<<SYS>>\n{prompt_system}\n<</SYS>>\n\n{prompt_content}"
        else:
            prompt = f"<<SYS>>\n{prompt_system}\n<</SYS>>\n\n{prompt_content}"

    return prompt_content
