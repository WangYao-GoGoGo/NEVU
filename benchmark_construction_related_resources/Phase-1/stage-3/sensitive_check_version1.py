"""Single-process sensitive-content checking script for Phase-1 filtering."""

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



def sensitive_check(client, content):
    response = client.moderations.create(
      model="omni-moderation-latest",
      input=content,
    )
    return response


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

    initial_dataset_file_path = initial_dataset_root_path + file_name  + ".json"
    datas = file_utils.read_json_file(initial_dataset_file_path)
    para_se = datas[process_start:process_end]
    hv_value_names_str = prompt_utils.get_hv_values(hv_path)
    prompt_content = file_utils.python_file_to_json(prompt_file_path)
    prompt_contents = prompt_utils.build_prompt(para_se, prompt_content, hv_value_names_str)
    client_check = openai_utils.open_api_instantiate(config.api_key)
    sens_file_root_path = output_root_path + file_name + "/sens_check/" + api_model + "/"

    for idx, prompt_con in enumerate(prompt_contents):
        local_guid = prompt_con["guid"]
        content = prompt_con["task_prompt"]
        completion = sensitive_check(client_check, content)
        comp_categories = completion.results[0].categories
        comp_category_scores = completion.results[0].category_scores
        file_index = preprocess_utils.compute_subfile_index(idx, process_start, max_rows_per_file)
        output_file_path = sens_file_root_path + file_index + "/"
        file_utils.check_and_create_file(output_file_path)
        output_file = output_file_path + "sens_result.json"
        sens_results = {
                        "guid": local_guid,
                        "categories": {
                            "harassment": {
                                "flag": comp_categories.harassment,
                                "confidence": comp_category_scores.harassment
                            },
                            "harassment_threatening":{
                                "flag": comp_categories.harassment_threatening,
                                "confidence": comp_category_scores.harassment_threatening
                            },
                            "hate":{
                                "flag": comp_categories.hate,
                                "confidence": comp_category_scores.hate
                            },
                            "hate_threatening":{
                                "flag": comp_categories.hate_threatening,
                                "confidence": comp_category_scores.hate_threatening
                            },
                            "illicit":{
                                "flag": comp_categories.illicit,
                                "confidence": comp_category_scores.illicit
                            },
                            "illicit_violent":{
                                "flag": comp_categories.illicit_violent,
                                "confidence": comp_category_scores.illicit_violent
                            },
                            "self_harm":{
                                "flag": comp_categories.self_harm,
                                "confidence": comp_category_scores.self_harm
                            },
                            "self_harm_instructions":{
                                "flag": comp_categories.self_harm_instructions,
                                "confidence": comp_category_scores.self_harm_instructions
                            },
                            "self_harm_intent":{
                                "flag": comp_categories.self_harm_intent,
                                "confidence": comp_category_scores.self_harm_intent
                            },
                            "sexual":{
                                "flag": comp_categories.sexual,
                                "confidence": comp_category_scores.sexual
                            },
                            "sexual_minors":{
                                "flag": comp_categories.sexual_minors,
                                "confidence": comp_category_scores.sexual_minors
                            },
                            "violence":{
                                "flag": comp_categories.violence,
                                "confidence": comp_category_scores.violence
                            },
                            "violence_graphic":{
                                "flag": comp_categories.violence_graphic,
                                "confidence": comp_category_scores.violence_graphic
                            }
                        }
                    }
        file_utils.append_json_data_to_file(output_file, sens_results, 4)
        print(completion)
