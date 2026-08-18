"""File and JSON helpers used across construction and experiment scripts."""

import os
import json
import sys
import time
from utils import data_utils
from typing import Any, Dict, List, Tuple, Set, Optional

def python_file_to_json(input_file):
    with open(input_file, 'r') as python_file:
        python_code = python_file.read()
    return python_code

def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        json_datas = json.load(json_file)
    return json_datas

def check_path_exist(file_path):
    if not os.path.exists(file_path):
        return False
    else:
        return True

def check_and_create_file(file_path):
    if not check_path_exist(file_path):
        os.makedirs(file_path)

def check_json_file_empty(file_path):
    with open(file_path, 'r') as data_file:
        file_content = data_file.read()
        if len(file_content) > 0:
            return False, file_content
        else:
            return True, file_content

def dump_json_file(file_path, json_data, indent):
    with open(file_path, 'w') as dump_file:
        json.dump(json_data, dump_file, indent=indent)

def find_max_id(processed_human_values):
    max_id = 0
    for processed_human_value in processed_human_values:
        id = processed_human_value["id"]
        if max_id < int(id):
            max_id = int(id)
    # max_id +=1
    return max_id

def check_exist(actor, human_value_label, processed_human_values):
    for processed_human_value in processed_human_values:
        processed_actor = processed_human_value["actor"]
        processed_human_value_label = processed_human_value["human_value_label"]
        if actor == processed_actor and human_value_label == processed_human_value_label:
            return True
    return False

def combination_human_values(unprocessed_human_values, processed_human_values):
    max_id = find_max_id(processed_human_values)
    for unprocessed_human_value in unprocessed_human_values:
        actor = unprocessed_human_value["actor"]
        human_value_label = unprocessed_human_value["human_value_label"]
        removal_impact_decision = unprocessed_human_value["removal_impact_decision"]
        confidence = unprocessed_human_value["confidence"]
        rationale = unprocessed_human_value["rationale"]
        check_result = check_exist(actor, human_value_label, processed_human_values)
        if not check_result:
            max_id = max_id + 1
            unprocessed_human_value["id"] = str(max_id)
            processed_human_values.append(unprocessed_human_value)
    return processed_human_values

def append_unprocessed_json_data_to_file_v2(processed_file_path, cur_json_object, indent, search_key):
    current_assessment_results = cur_json_object["assessment_results"]
    guid = cur_json_object["guid"]
    if check_path_exist(processed_file_path):
        result, file_content = check_json_file_empty(processed_file_path)
        if result:
            processed_cur_json_objects = [cur_json_object]
        else:
            processed_cur_json_objects = json.loads(file_content)
            exist = False
            for current_assessment_result in current_assessment_results:
                current_search_value = current_assessment_result[search_key]
                related_processed_assessment_results = data_utils.find_item_by_key(current_search_value, search_key,
                                                                         processed_cur_json_objects)
                if related_processed_assessment_results:
                    if "aligned_with_human_values" in related_processed_assessment_results:
                        processed_aligned_with_human_values = related_processed_assessment_results["aligned_with_human_values"]
                        if "aligned_with_human_values" in current_assessment_result:
                            current_aligned_with_human_values = current_assessment_result[
                                "aligned_with_human_values"]
                        else:
                            current_aligned_with_human_values = []
                        processed_aligned_with_human_values = combination_human_values(
                            current_aligned_with_human_values, processed_aligned_with_human_values)
                    else:
                        processed_aligned_with_human_values = []
                    if "contradictory_to_human_values" in related_processed_assessment_results:
                        processed_contradictory_to_human_values = related_processed_assessment_results[
                            "contradictory_to_human_values"]
                        if "contradictory_to_human_values" in current_assessment_result:
                            current_contradictory_to_human_values = current_assessment_result[
                                "contradictory_to_human_values"]
                        else:
                            processed_contradictory_to_human_values = []
                        processed_contradictory_to_human_values = combination_human_values(
                            current_contradictory_to_human_values,
                            processed_contradictory_to_human_values)
                    else:
                        processed_contradictory_to_human_values = []
            # for processed_cur_json_object in processed_cur_json_objects:
            #     processed_guid = processed_cur_json_object["guid"]
            #     if guid == processed_guid:
            #         exist = True
            #         processed_assessment_results = processed_cur_json_object["assessment_results"]
            #         for processed_assessment_result in processed_assessment_results:
            #             processed_search_value = processed_assessment_result[search_key]
            #             related_assessment_results = data_utils.find_item_by_key(processed_search_value, search_key, assessment_results)
            #             if related_assessment_results:
            #                 if "aligned_with_human_values" in related_assessment_results:
            #                     unprocessed_aligned_with_human_values = related_assessment_results["aligned_with_human_values"]
            #                     if "aligned_with_human_values" in processed_assessment_result:
            #                         processed_aligned_with_human_values = processed_assessment_result[
            #                             "aligned_with_human_values"]
            #                     else:
            #                         processed_aligned_with_human_values = []
            #                     processed_aligned_with_human_values = combination_human_values(
            #                             unprocessed_aligned_with_human_values, processed_aligned_with_human_values)
            #                 else:
            #                     unprocessed_aligned_with_human_values = []
            #                 if "contradictory_to_human_values" in related_assessment_results:
            #                     unprocessed_contradictory_to_human_values = related_assessment_results["contradictory_to_human_values"]
            #                     if "contradictory_to_human_values" in processed_assessment_result:
            #                         processed_contradictory_to_human_values = processed_assessment_result[
            #                             "contradictory_to_human_values"]
            #                     else:
            #                         processed_contradictory_to_human_values = []
            #                     processed_contradictory_to_human_values = combination_human_values(
            #                         unprocessed_contradictory_to_human_values,
            #                         processed_contradictory_to_human_values)
            #                 else:
            #                     unprocessed_contradictory_to_human_values = []
                    # break
            if not exist:
                processed_cur_json_objects.append(cur_json_object)
    else:
        processed_cur_json_objects = [cur_json_object]
    # update data
    print("save data!")
    dump_json_file(processed_file_path, processed_cur_json_objects, indent)

def append_unprocessed_json_data_to_file(processed_file_path, cur_json_object, indent, search_key):
    assessment_results = cur_json_object["assessment_results"]
    guid = cur_json_object["guid"]
    if check_path_exist(processed_file_path):
        result, file_content = check_json_file_empty(processed_file_path)
        if result:
            processed_cur_json_objects = [cur_json_object]
        else:
            processed_cur_json_objects = json.loads(file_content)
            exist = False
            for processed_cur_json_object in processed_cur_json_objects:
                processed_guid = processed_cur_json_object["guid"]
                if guid == processed_guid:
                    exist = True
                    processed_assessment_results = processed_cur_json_object["assessment_results"]
                    for processed_assessment_result in processed_assessment_results:
                        processed_search_value = processed_assessment_result[search_key]
                        related_assessment_results = data_utils.find_item_by_key(processed_search_value, search_key, assessment_results)
                        if related_assessment_results:
                            if "aligned_with_human_values" in related_assessment_results:
                                unprocessed_aligned_with_human_values = related_assessment_results["aligned_with_human_values"]
                                if "aligned_with_human_values" in processed_assessment_result:
                                    processed_aligned_with_human_values = processed_assessment_result[
                                        "aligned_with_human_values"]
                                else:
                                    processed_aligned_with_human_values = []
                                processed_aligned_with_human_values = combination_human_values(
                                        unprocessed_aligned_with_human_values, processed_aligned_with_human_values)
                            else:
                                unprocessed_aligned_with_human_values = []
                            if "contradictory_to_human_values" in related_assessment_results:
                                unprocessed_contradictory_to_human_values = related_assessment_results["contradictory_to_human_values"]
                                if "contradictory_to_human_values" in processed_assessment_result:
                                    processed_contradictory_to_human_values = processed_assessment_result[
                                        "contradictory_to_human_values"]
                                else:
                                    processed_contradictory_to_human_values = []
                                processed_contradictory_to_human_values = combination_human_values(
                                    unprocessed_contradictory_to_human_values,
                                    processed_contradictory_to_human_values)
                            else:
                                unprocessed_contradictory_to_human_values = []
                    # break
            if not exist:
                processed_cur_json_objects.append(cur_json_object)
    else:
        processed_cur_json_objects = [cur_json_object]
    # update data
    print("save data!")
    dump_json_file(processed_file_path, processed_cur_json_objects, indent)

def append_json_data_to_file(file_path, cur_json_object, indent):
    if check_path_exist(file_path):
        result, file_content = check_json_file_empty(file_path)
        if result:
            cur_json_objects = [cur_json_object]
        else:
            cur_json_objects = json.loads(file_content)
            cur_json_objects.append(cur_json_object)
    else:
        cur_json_objects = [cur_json_object]
    dump_json_file(file_path, cur_json_objects, indent)

def write_txt_file(file_path, txt_data):
    with open(file_path, "a") as file:
        file.write(txt_data + "\r\n")

# ----------------------------
# Utilities: JSON IO
# ----------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
