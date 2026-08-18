"""Compute human-value label distributions across Phase-1 subfiles."""

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
    file_list = list(range(file_start, file_end))
    datas_subfiles = {}

    hv_count = {}
    hv_guids = {}
    for file in file_list:
        completed_articles = []
        uncompleted_subfiles = []
        initial_dataset_file_path = subfiles_root_path + "part_" + str(file)  + ".json"
        completed_file_path = completed_root_path + str(file) + "/" + config.completed_filename
        completed_subfiles = file_utils.read_json_file(completed_file_path)
        for completed_article in completed_subfiles:
            article_infos = json.loads(completed_article)
            guid = article_infos["guid"]
            hv_categories_dic = article_infos["hv_categories"]
            sorted_data = sorted(hv_categories_dic.items(), key=lambda x: x[1], reverse=True)
            hv_categories = [k for k, v in sorted_data[:1]] # top n

            for hv_categorie in hv_categories:
                if hv_categorie in few_types:
                    if hv_categorie not in hv_count.keys():
                        hv_count[hv_categorie] = 1
                    else:
                        hv_count[hv_categorie] = hv_count[hv_categorie] + 1
                    completed_articles.append(completed_article)
                if file in hv_guids.keys():
                    hv_guid = hv_guids[file]
                    if hv_categorie in hv_guid.keys():
                        hv_guid[hv_categorie].append(guid)
                    else:
                        guid_values = []
                        guid_values.append(guid)
                        hv_guid[hv_categorie] = guid_values
                else:
                    hv_guid = {}
                    guid_values = []
                    guid_values.append(guid)
                    hv_guid[hv_categorie] = guid_values
                    hv_guids[file] = hv_guid
        datas_subfiles[file] = completed_articles
    # Save datas_subfiles
    # file_utils.dump_json_file(os.path.join(output_root_path, "datas_subfiles.json"), datas_subfiles)
    # Save hv_count
    file_utils.dump_json_file(os.path.join(output_root_path, "hv_count.json"), hv_count, 4)
    # Save hv_guids
    file_utils.dump_json_file(os.path.join(output_root_path, "hv_guids.json"), hv_guids, 4)
    print("finished")
