"""Combine sensitive-content checking outputs across Phase-1 subfiles."""

from openai import OpenAI
import os
import json
import sys
import time
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import config.config_multithread as config
from utils import file_utils
from utils import openai_utils
from utils import preprocess_utils
from utils import prompt_utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=False, help="API key for accessing the service")
    parser.add_argument("--config", type=str, default="/tmp/echv_wangyao/config/echv_config_sens_check.json")

    # ************TEST************
    # args = parser.parse_args(["--api_key", ""])
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
    output_filename = config.output_filename

    file_list = list(range(file_start, file_end))
    file_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
    # file_list = ["1"]
    output_file_root_path = output_root_path + "output/" + api_model + "/"
    datas_subfiles = []
    for file in file_list:
        initial_dataset_file_path = output_file_root_path + str(file) + "/"  + output_filename
        datas_subfile = file_utils.read_json_file(initial_dataset_file_path)
        if preprocess_utils.str_to_bool(is_process):
            datas_subfile = datas_subfile[process_start: process_end]
        datas_subfiles.extend(datas_subfile)

    file_utils.check_and_create_file(output_file_root_path + "combination/")
    output_file_combination_root_path = output_file_root_path + "combination/" + "sens_check_results.json"
    file_utils.append_json_data_to_file(output_file_combination_root_path, datas_subfiles, 4)

    print("finished!")
