"""Prepare per-file statistics and processing ranges for event extraction."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import config.config_statistics_event_extraction as config
from utils import file_utils
from dataset.wiki_ontology import event_genres

def data_statistics(file_name, config):
    output_root_path = config.output_root_path + "initial_dataset_" + file_name
    output_article_genres_file = output_root_path + "/article_genres.json"
    json_data = file_utils.read_json_file(output_article_genres_file)
    return json_data

def count_plus(statistics_results, genre):
    if statistics_results.__contains__(genre):
        statistics_results[genre] = statistics_results[genre] + 1
    else:
        statistics_results[genre] = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=False, help="API key for accessing the service")
    parser.add_argument("--config", type=str, default="/tmp/echv_preprocessing/config/echv_config_statistics_event_extraction.json")

    # ************TEST************
    # args = parser.parse_args(["--api_key", "123"])
    args = parser.parse_args()
    # ************TEST************

    config = config.Config(args)
    initial_dataset_root_path = config.initial_dataset_root_path
    prompt_file_path = config.prompt_file_path
    hv_path = config.hv_path
    output_root_path = config.output_root_path + config.api_model + "/"
    process_start = config.process_start
    process_end = config.process_end
    file_start = config.file_start
    file_end = config.file_end
    max_rows_per_file = config.max_rows_per_file
    subfiles_root_path = config.subfiles_root_path
    min_thread_count = config.min_thread_count
    is_process = config.is_process

    file_list = list(range(file_start, file_end))
    # file_list = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    output_files = {}
    total_count = 0
    for file in file_list:
        output_file_path = output_root_path + str(file) + "/" + "output.json"
        output_file = file_utils.read_json_file(output_file_path)
        all_subevents = []
        # abc = [subevent["sub_events"][0] for subevent in output_file]
        for subevents in output_file:
            sub_event = subevents["sub_events"]
            all_subevents.extend(sub_event)
        total_count = total_count + len(all_subevents)
        output_files[file] = all_subevents
    files = output_files.items()
    print("total count: " + str(total_count))
