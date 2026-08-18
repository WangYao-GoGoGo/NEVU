"""Multithreaded sensitive-content checking script for Phase-1 filtering."""

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
from utils import llm_utils
from utils import preprocess_utils
from utils import prompt_utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

def process_and_store(filename, data_list):
    processed_count = 0
    print(f"Process ID: {os.getpid()}, Thread ID: {threading.get_ident()}, filename: {filename}\n")
    for idx, prompt_con in enumerate(data_list):
        local_guid = prompt_con["guid"]
        content = prompt_con["content"]
        # empty_article = prompt_con["empty_article"]
        print("data_idx: " + str(idx) + "\n")
        try:
            start_time = time.time()
            if content:
                completion = llm_utils.sensitive_check(client, content)
            end_time = time.time()
            hours, minutes, seconds = preprocess_utils.spent_time(start_time, end_time)

            # ************TEST************
            # args = parser.parse_args()
            log_con = "*******current file name: " + str(filename) + ",start:" + str(process_start) + ", end: " + str(
                process_end) + ", api_model:" + api_model + ", current num: " + str(process_start +
                processed_count) + f", SpentTimeforPerArticle: {hours}h {minutes}m {seconds:.2f}s" + "********"
            print(log_con)
            # ************TEST************
            # file_index = preprocess_utils.compute_subfile_index(idx, process_start, max_rows_per_file)

            output_file = output_file_root_path + str(filename) + "/"

            file_utils.check_and_create_file(output_file)

            output_file_name = output_file + config.output_filename

            comp_categories = completion.results[0].categories
            comp_category_scores = completion.results[0].category_scores
            file_index = preprocess_utils.compute_subfile_index(idx, process_start, max_rows_per_file)
            # output_file_path = output_root_path + file_index + "/"
            file_utils.check_and_create_file(output_file)
            sens_results = {
                "guid": local_guid,
                "categories": {
                    "harassment": {
                        "flag": comp_categories.harassment,
                        "confidence": comp_category_scores.harassment
                    },
                    "harassment_threatening": {
                        "flag": comp_categories.harassment_threatening,
                        "confidence": comp_category_scores.harassment_threatening
                    },
                    "hate": {
                        "flag": comp_categories.hate,
                        "confidence": comp_category_scores.hate
                    },
                    "hate_threatening": {
                        "flag": comp_categories.hate_threatening,
                        "confidence": comp_category_scores.hate_threatening
                    },
                    "illicit": {
                        "flag": comp_categories.illicit,
                        "confidence": comp_category_scores.illicit
                    },
                    "illicit_violent": {
                        "flag": comp_categories.illicit_violent,
                        "confidence": comp_category_scores.illicit_violent
                    },
                    "self_harm": {
                        "flag": comp_categories.self_harm,
                        "confidence": comp_category_scores.self_harm
                    },
                    "self_harm_instructions": {
                        "flag": comp_categories.self_harm_instructions,
                        "confidence": comp_category_scores.self_harm_instructions
                    },
                    "self_harm_intent": {
                        "flag": comp_categories.self_harm_intent,
                        "confidence": comp_category_scores.self_harm_intent
                    },
                    "sexual": {
                        "flag": comp_categories.sexual,
                        "confidence": comp_category_scores.sexual
                    },
                    "sexual_minors": {
                        "flag": comp_categories.sexual_minors,
                        "confidence": comp_category_scores.sexual_minors
                    },
                    "violence": {
                        "flag": comp_categories.violence,
                        "confidence": comp_category_scores.violence
                    },
                    "violence_graphic": {
                        "flag": comp_categories.violence_graphic,
                        "confidence": comp_category_scores.violence_graphic
                    }
                }
            }
            # current news
            file_utils.append_json_data_to_file(output_file_name, sens_results, 4)

            processed_count += 1
            print("guid:" + local_guid + " is processed!")
        except Exception as e:
            print(f"Error result: {e}")
            continue

def thread_processing(prompt_cons_file, min_thread_count):
    tasks = list(prompt_cons_file.items())

    tasks = list(prompt_cons_file.items())
    max_workers = min(min_thread_count, cpu_count())
    print(f"CPU cores: {cpu_count()}, threads: {max_workers}")

    # Run tasks with multiple threads
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_task = {
            executor.submit(process_and_store, filename, data_list): (filename, data_list)
            for filename, data_list in tasks
        }

        # Wait for tasks to finish and collect results
        for future in as_completed(future_to_task):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print("Error while executing task:", e)

    # # Print or process returned results
    # for res in results:
    #     print("Completed ->", res)
    #
    # with Pool(processes=min(min_thread_count, cpu_count())) as pool:
    #     print(cpu_count())
    #     results = pool.starmap(process_and_store, tasks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=False, help="API key for accessing the service")
    parser.add_argument("--config", type=str, default="/tmp/echv_wangyao/config/echv_config_sens_check.json")

    # ************TEST************
    api_key = os.getenv('api_key')
    args = parser.parse_args(["--api_key", api_key])
    # ************formal************
    # args = parser.parse_args()
    # ****************************

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
    incre_data_path = config.incre_data_path

    file_list = list(range(file_start, file_end))

    # file_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    # file_list = ["1"]
    datas_subfiles = {}
    exe_type = "p1_v2"
    for file in file_list:
        if exe_type == "p1_v1":
            tmp_incre_data_path = subfiles_root_path + "part_" + str(file)  + ".json"
        else:
            tmp_incre_data_path = incre_data_path + str(file) + "/event_base_with_actors.json"
        datas_subfile = file_utils.read_json_file(tmp_incre_data_path)
        if preprocess_utils.str_to_bool(is_process):
            datas_subfile = datas_subfile[process_start: process_end]
        datas_subfiles[file] = datas_subfile
        # datas_subfiles.append(datas_subfile)

    # para_se = datas[procezss_start:process_end]
    # hv_value_names_str = prompt_utils.get_hv_values(hv_path)
    # prompt_content = file_utils.python_file_to_json(prompt_file_path)
    # prompt_cons_file = prompt_utils.build_sens_prompt_multithread(datas_subfiles, prompt_content, hv_value_names_str)

    # ************TEST************
    # file_data = {}
    # for idx, prompt_con in enumerate(prompt_contents):
    #     file_index = preprocess_utils.compute_subfile_index(idx, process_start, file_data, max_rows_per_file)
    #     print(str(file_index))
    # ************TEST************

    client = llm_utils.open_api_instantiate(config.api_key)

    # ************TEST************
    # output_file = output_root_path + file_name + "/output_test/" + api_model + "/"
    # output_completed_file = output_root_path + file_name + "/output_completed_test/" + api_model + "/"
    # res_content_file_path = output_root_path + file_name + "/res_test/" + api_model + "/"
    # ************TEST************

    # ************formal************
    output_file_root_path = output_root_path + "phase1_v2/output/" + api_model + "/"
    # ************formal************
    min_thread_count = len(datas_subfiles)
    thread_processing(datas_subfiles, min_thread_count)
