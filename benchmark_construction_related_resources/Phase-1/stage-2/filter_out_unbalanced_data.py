import os
import json
import sys
import time
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import config.config_filter_out_unbalanced_data as config
from utils import file_utils
from utils import preprocess_utils
from dataset.unbalanced_distribution.few_types import few_types
from dataset.unbalanced_distribution.few_types_count import few_types_count
from collections import defaultdict

def newly_added_label_count(newly_added_files):
    added_hv_counts = defaultdict(int)
    for file_id, hv_dict in newly_added_files.items():
        for hv_label, guid_list in hv_dict.items():
            added_hv_counts[hv_label] += len(guid_list)

    # 打印统计结果
    print(" 各 few_hv_label 的补充数量：")
    for hv_label, count in added_hv_counts.items():
        print(f" - {hv_label}: {count}")
    return added_hv_counts

def collect_newly_added_guids(few_types_count, file_hv_guids):
    newly_added_files = {}
    for few_hv_label, hv_count in few_types_count.items():
        newly_added_count = fixed_count - hv_count
        if newly_added_count <= 0:
            continue  # 已经够了，不需要补充

        loop_count = newly_added_count
        for file_id, hv_items in file_hv_guids.items():
            if few_hv_label in hv_items:
                guids = hv_items[few_hv_label]
                if not guids:
                    continue

                selected_guids = guids[:loop_count]  # 最多取 loop_count 个
                loop_count -= len(selected_guids)

                # 初始化并追加
                if file_id not in newly_added_files:
                    newly_added_files[file_id] = {}
                if few_hv_label not in newly_added_files[file_id]:
                    newly_added_files[file_id][few_hv_label] = []

                newly_added_files[file_id][few_hv_label].extend(selected_guids)

                if loop_count <= 0:
                    break  # 补充够了，就停止遍历
    print("newly_added_files finished!")
    return newly_added_files

def filter_out_unbalanced_data(file_path):
    hv_guids = file_utils.read_json_file(file_path)
    # 遍历结构：文件号 -> human value -> guid 列表
    file_hv_guids = {}
    for file_id, value_dict in hv_guids.items():
        print(f"文件编号: {file_id}")
        filtered_hv_guids = {}
        for hv_type, guid_list in value_dict.items():
            print(f"  - human value: {hv_type}")
            filtered_guids = []
            if hv_type in few_types:
                filtered_guids.append(guid_list)
            if filtered_guids:
                filtered_hv_guids[hv_type] = filtered_guids
        if filtered_guids:
            file_hv_guids[file_id] = filtered_hv_guids
    print("file_hv_guids finished!")
    return file_hv_guids

def compute_rest_labels(newly_added_files, few_types_count):
    added_hv_counts = defaultdict(int)
    for file_id, hv_dict in newly_added_files.items():
        for hv_label, guid_list in hv_dict.items():
            added_hv_counts[hv_label] += len(guid_list)

    # 2. 整合原始数量与新增数量，得到实际总量
    final_counts = {}
    for hv_label, original_count in few_types_count.items():
        added_count = added_hv_counts.get(hv_label, 0)
        final_counts[hv_label] = original_count + added_count

    # 3. 筛选出仍未达 100 的标签
    underfilled_labels = {label: count for label, count in final_counts.items() if count < 100}
    still_needed_labels = {label: 100 - count for label, count in final_counts.items() if count < 100}

    # 4. 打印结果
    print(" 以下 few_types 中仍不足 100 个 guid 的标签：")
    for label, count in underfilled_labels.items():
        print(f" - {label}: 当前总数 {count}")
    return underfilled_labels, still_needed_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/tmp/echv_preprocessing/config/echv_config_filter_out_unbalanced_data.json")

    # ************TEST************
    args = parser.parse_args()
    # ************TEST************

    config = config.Config(args)
    initial_dataset_root_path = config.initial_dataset_root_path
    prompt_file_path = config.prompt_file_path
    hv_path = config.hv_path
    role1 = config.role1
    role2 = config.role2
    content1 = config.content1
    output_root_path = config.output_root_path
    process_start = config.process_start
    process_end = config.process_end
    file_start = config.file_start
    file_end = config.file_end
    api_model = config.api_model
    max_rows_per_file = config.max_rows_per_file
    subfiles_root_path = config.subfiles_root_path
    min_thread_count = config.min_thread_count
    is_process = config.is_process
    completed_root_path = config.completed_output_root_path

    file_utils.check_and_create_file(output_root_path)
    # output_old_file_path = output_root_path + api_model + "/"
    file_list = list(range(file_start, file_end))
    datas_subfiles = {}

    hv_count = {}
    hv_guids = {}
    hv_guids_file_path = "dataset/statistics/hv_guids.json"
    file_hv_guids = filter_out_unbalanced_data(hv_guids_file_path)
    # fixed_count = 100
    # newly_added_files = {}
    # for few_hv_label, hv_count in few_types_count.items():
    #     newly_added_count = fixed_count - hv_count
    #     loop_count = newly_added_count
    #     # for 0 in newly_added_count:
    #     last_flag = False
    #     while loop_count > 0 and not last_flag:
    #         newly_added_hvs = {}
    #         items = list(file_hv_guids.items())
    #         for index, (file_id, hv_items) in enumerate(items):
    #             new_guids = []
    #             for hv_label, guids in hv_items.items():
    #                 if few_hv_label == hv_label:
    #                     for guid in guids:
    #                         if loop_count != 0:
    #                             new_guids.append(guid)
    #                             loop_count = loop_count - 1
    #             if index == len(items) - 1:
    #                 last_flag = True
    #             if new_guids:
    #                newly_added_hvs[few_hv_label] = new_guids
    #         if newly_added_hvs:
    #             newly_added_files[file_id] = newly_added_hvs
    # print("finished!")

    fixed_count = 100
    newly_added_files = collect_newly_added_guids(few_types_count, file_hv_guids)
    added_hv_counts = newly_added_label_count(newly_added_files)

    underfilled_labels, still_needed_labels = compute_rest_labels(newly_added_files, few_types_count)

    print("all tasks finished!")

