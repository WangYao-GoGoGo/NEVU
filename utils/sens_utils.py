from openai import OpenAI

import os
import json
import sys
import time

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)

import argparse
import config.config as config
from formal_extraction.utils import file_utils
from formal_extraction.utils import openai_utils
from formal_extraction.utils import preprocess_utils
from formal_extraction.utils import prompt_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=False, help="API key for accessing the service")
    parser.add_argument("--config", type=str, default="/tmp/echv_wangyao/config/echv_config_sens_check.json")

    # ************TEST************
    # args = parser.parse_args(["--api_key", "123"])
    args = parser.parse_args()
    # ************TEST************

    config = config.Config(args)
    file_name = config.file_name
    initial_dataset_root_path = config.initial_dataset_root_path
    prompt_file_path = config.prompt_file_path
    hv_path = config.hv_path
    role1 = config.role1
    role2 = config.role2
    content1 = config.content1
    output_root_path = config.output_root_path
    process_start = config.process_start
    process_end = config.process_end
    api_model = config.api_model
    max_rows_per_file = config.max_rows_per_file

    file_index = "0"
    sens_file_root_path = output_root_path + file_name + "/sens_check/" + api_model + "/"
    output_file_path = sens_file_root_path + file_index + "/"
    output_file = output_file_path + "sens_result.json"

    sens_results = file_utils.read_json_file(output_file)
    # problem_datas = []

    # 筛选出 flag 为 True 的数据
    filtered_data = []

    for item in sens_results:
        guid = item["guid"]
        categories = item["categories"]
        filtered_categories = {
            category: details
            for category, details in categories.items()
            if details["flag"]  # 筛选出 flag 为 True 的类别
        }
        if filtered_categories:  # 如果有筛选结果，加入最终结果
            filtered_data.append({"guid": guid, "categories": filtered_categories})

    print(json.dumps(filtered_data, indent=4))

    # for sens_result in sens_results:
    #     sens_categories = sens_result["categories"]
    #     harassment_flag = sens_categories["harassment"]["flag"]
    #     harassment_threatening_flag = sens_categories["harassment_threatening"]["flag"]
    #     hate_flag = sens_categories["hate"]["flag"]
    #     hate_threatening_flag = sens_categories["hate_threatening"]["flag"]
    #     illicit_flag = sens_categories["illicit"]["flag"]
    #     illicit_violent_flag = sens_categories["illicit_violent"]["flag"]
    #     self_harm_flag = sens_categories["self_harm"]["flag"]
    #     self_harm_instructions_flag = sens_categories["self_harm_instructions"]["flag"]
    #     self_harm_intent_flag = sens_categories["self_harm_intent"]["flag"]
    #     sexual_flag = sens_categories["sexual"]["flag"]
    #     sexual_minors_flag = sens_categories["sexual_minors"]["flag"]
    #     violence_flag = sens_categories["violence"]["flag"]
    #     violence_graphic_flag = sens_categories["violence_graphic"]["flag"]

    # print(sens_result)
