import os
import json
import sys
import time
import glob
from collections import defaultdict

def count_human_values(files, hv_value_desc_dict):
    total_counts = {
        "article_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)},
        "subevents_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)},
        "behavior_chains_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)},
        "story_narrative_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)}
    }

    for data in files:
            for section in total_counts:
                # print(str(section))
                for value_type in total_counts[section]:
                    if section not in data.keys():
                        continue
                    if section in ("subevents_human_values", "behavior_chains_human_values", "story_narrative_human_values"):
                        contents_human_values = data[section]
                        for content_human_values in contents_human_values:
                            if value_type not in content_human_values.keys():
                                continue
                            for label, details in content_human_values[value_type].items():
                                if label == "Be safe country":
                                    label = "Have a safe country"
                                if label not in hv_value_desc_dict:
                                    continue
                                total_counts[section][value_type][label] += 1
                                level2 = hv_value_desc_dict[label]["level-2"]
                                levels3 = hv_value_desc_dict[label]["level-3"]
                                levels4a = hv_value_desc_dict[label]["level-4a"]
                                levels4b = hv_value_desc_dict[label]["level-4b"]
                                total_counts[section][value_type][level2] += 1
                                for level3 in levels3:
                                    total_counts[section][value_type]["level-3"][level3] += 1
                                for level4a in levels4a:
                                    total_counts[section][value_type]["level-4a"][level4a] += 1
                                for level4b in levels4b:
                                    total_counts[section][value_type]["level-4b"][level4b] += 1
                    else:
                        for label, details in data[section][value_type].items():
                            if label == "Be safe country":
                                label = "Have a safe country"
                            if label not in hv_value_desc_dict:
                                continue
                            total_counts[section][value_type][label] += 1
                            level2 = hv_value_desc_dict[label]["level-2"]
                            levels3 = hv_value_desc_dict[label]["level-3"]
                            levels4a = hv_value_desc_dict[label]["level-4a"]
                            levels4b = hv_value_desc_dict[label]["level-4b"]
                            total_counts[section][value_type][level2] += 1
                            for level3 in levels3:
                                total_counts[section][value_type]["level-3"][level3] += 1
                            for level4a in levels4a:
                                total_counts[section][value_type]["level-4a"][level4a] += 1
                            for level4b in levels4b:
                                total_counts[section][value_type]["level-4b"][level4b] += 1

    # 转换defaultdict为普通dict以便输出
    for section in total_counts:
        for value_type in total_counts[section]:
            total_counts[section][value_type] = dict(total_counts[section][value_type])

    return total_counts

def count_human_values_plus(files, hv_value_desc_dict):
    total_counts = {
        "article_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)},
        "subevents_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)},
        "behavior_chains_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)},
        "story_narrative_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)}
    }

    # 单独初始化level2, level3, level4a, level4b
    level_counts = {}
    for section in total_counts:
        level_counts[section] = {}
        for value_type in total_counts[section]:
            level_counts[section][value_type] = {
                "level2": defaultdict(int),
                "level3": defaultdict(int),
                "level4a": defaultdict(int),
                "level4b": defaultdict(int)
            }

    for data in files:
        for section in total_counts:
            for value_type in total_counts[section]:
                if section not in data:
                    continue
                if section in ("subevents_human_values", "behavior_chains_human_values", "story_narrative_human_values"):
                    contents_human_values = data[section]
                    for content_human_values in contents_human_values:
                        if value_type not in content_human_values:
                            continue
                        for label in content_human_values[value_type]:
                            label_fixed = "Have a safe country" if label == "Be safe country" else label
                            if label_fixed not in hv_value_desc_dict:
                                continue
                            total_counts[section][value_type][label_fixed] += 1
                            level2 = hv_value_desc_dict[label_fixed]["level-2"]
                            levels3 = hv_value_desc_dict[label_fixed]["level-3"]
                            levels4a = hv_value_desc_dict[label_fixed]["level-4a"]
                            levels4b = hv_value_desc_dict[label_fixed]["level-4b"]

                            level_counts[section][value_type]["level-2"][level2] += 1
                            for level3 in levels3:
                                level_counts[section][value_type]["level-3"][level3] += 1
                            for level4a in levels4a:
                                level_counts[section][value_type]["level-4a"][level4a] += 1
                            for level4b in levels4b:
                                level_counts[section][value_type]["level-4b"][level4b] += 1
                else:
                    for label in data[section][value_type]:
                        label_fixed = "Have a safe country" if label == "Be safe country" else label
                        if label_fixed not in hv_value_desc_dict:
                            continue
                        total_counts[section][value_type][label_fixed] += 1
                        level2 = hv_value_desc_dict[label_fixed]["level-2"]
                        levels3 = hv_value_desc_dict[label_fixed]["level-3"]
                        levels4a = hv_value_desc_dict[label_fixed]["level-4a"]
                        levels4b = hv_value_desc_dict[label_fixed]["level-4b"]

                        level_counts[section][value_type]["level-2"][level2] += 1
                        for level3 in levels3:
                            level_counts[section][value_type]["level-3"][level3] += 1
                        for level4a in levels4a:
                            level_counts[section][value_type]["level-4a"][level4a] += 1
                        for level4b in levels4b:
                            level_counts[section][value_type]["level-4b"][level4b] += 1

    # 转换defaultdict为普通dict以便输出
    for section in total_counts:
        for value_type in total_counts[section]:
            total_counts[section][value_type] = dict(total_counts[section][value_type])
            level_counts[section][value_type] = {k: dict(v) for k, v in level_counts[section][value_type].items()}

    return {"level1_counts": total_counts, "level_counts": level_counts}

def count_human_values_with_mislabels(files, hv_value_desc_dict):
    total_counts = {
        "article_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)},
        "subevents_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)},
        "behavior_chains_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)},
        "story_narrative_human_values": {"aligned_with_human_values": defaultdict(int), "contradictory_to_human_values": defaultdict(int)}
    }

    level_counts = {}
    missing_labels_a = defaultdict(set)
    missing_labels_b = defaultdict(set)

    for section in total_counts:
        level_counts[section] = {}
        for value_type in total_counts[section]:
            level_counts[section][value_type] = {
                "level-2": defaultdict(int),
                "level-3": defaultdict(int),
                "level-4a": defaultdict(int),
                "level-4b": defaultdict(int)
            }

    for data in files:
        for section in total_counts:
            for value_type in total_counts[section]:
                if section not in data:
                    continue
                if section in ("subevents_human_values", "behavior_chains_human_values", "story_narrative_human_values"):
                    contents_human_values = data[section]
                    for content_human_values in contents_human_values:
                        for actor, content_human_value in content_human_values.items():
                            if value_type not in content_human_value:
                                continue
                            for label in content_human_value[value_type]:
                                label_fixed = "Have a safe country" if label == "Be safe country" else label
                                if label_fixed not in hv_value_desc_dict:
                                    missing_labels_a[section].add(label_fixed)
                                    continue
                                total_counts[section][value_type][label_fixed] += 1
                                level2 = hv_value_desc_dict[label_fixed]["level-2"]
                                levels3 = hv_value_desc_dict[label_fixed]["level-3"]
                                levels4a = hv_value_desc_dict[label_fixed]["level-4a"]
                                levels4b = hv_value_desc_dict[label_fixed]["level-4b"]

                                level_counts[section][value_type]["level-2"][level2] += 1
                                for level3 in levels3:
                                    level_counts[section][value_type]["level-3"][level3] += 1
                                for level4a in levels4a:
                                    level_counts[section][value_type]["level-4a"][level4a] += 1
                                for level4b in levels4b:
                                    level_counts[section][value_type]["level-4b"][level4b] += 1
                else:
                    for actor, hv_labels in data[section].items():
                        if value_type in hv_labels:
                            for label in hv_labels[value_type]:
                                label_fixed = "Have a safe country" if label == "Be safe country" else label
                                if label_fixed not in hv_value_desc_dict:
                                    missing_labels_a[section].add(label_fixed)
                                    continue
                                total_counts[section][value_type][label_fixed] += 1
                                level2 = hv_value_desc_dict[label_fixed]["level-2"]
                                levels3 = hv_value_desc_dict[label_fixed]["level-3"]
                                levels4a = hv_value_desc_dict[label_fixed]["level-4a"]
                                levels4b = hv_value_desc_dict[label_fixed]["level-4b"]

                                level_counts[section][value_type]["level-2"][level2] += 1
                                for level3 in levels3:
                                    level_counts[section][value_type]["level-3"][level3] += 1
                                for level4a in levels4a:
                                    level_counts[section][value_type]["level-4a"][level4a] += 1
                                for level4b in levels4b:
                                    level_counts[section][value_type]["level-4b"][level4b] += 1

    # 转换default dict为普通dict以便输出
    for section in total_counts:
        for value_type in total_counts[section]:
            total_counts[section][value_type] = dict(total_counts[section][value_type])
            level_counts[section][value_type] = {k: dict(v) for k, v in level_counts[section][value_type].items()}

    missing_labels_a = {k: list(v) for k, v in missing_labels_a.items()}

    return {"level1_counts": total_counts, "level_counts": level_counts, "missing_labels_a": missing_labels_a}