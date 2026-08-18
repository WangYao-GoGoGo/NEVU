import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import config.config_statistics as config
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
    parser.add_argument("--config", type=str, default="/tmp/echv_preprocessing/config/echv_config_statistics.json")

    # ************TEST************
    # args = parser.parse_args(["--api_key", "123"])
    args = parser.parse_args()
    # ************TEST************

    config = config.Config(args)
    file_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]

    data_map = {}
    for file_name in file_list:
        sub_data = data_statistics(file_name, config)
        data_map[file_name] = sub_data

    event_genres = event_genres.genre
    statistics_results = {}
    for event_genre in event_genres:
        statistics_results[event_genre] = 0

    count = 0
    for file_name in data_map.keys():
        datas = data_map[file_name]
        for data in datas:
            guid = data["guid"]
            genre = data["label"]
            count_plus(statistics_results, genre)
            count = count + 1
    print("total count:" + str(count))
    print(str(statistics_results))