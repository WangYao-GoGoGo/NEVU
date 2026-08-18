"""Build balanced Phase-1 subfiles by distributing article GUIDs across genres."""

import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import config.config as config
from utils import file_utils
# from dataset.wiki_ontology import event_genres
from dataset.wiki_ontology import event_genres_without_sensdata

def data_statistics(file_name, config):
    output_root_path = config.output_root_path + "initial_dataset_" + file_name
    output_article_genres_file = output_root_path + "/article_genres.json"
    json_data = file_utils.read_json_file(output_article_genres_file)
    return json_data

def classify_data(genre_data, genre, guid, event_genres):
    if genre in event_genres:
        if genre_data.__contains__(genre):
            data_list = genre_data[genre]
            data_list.append(guid)
            genre_data[genre] = data_list
        else:
            data_list = [guid]
            genre_data[genre] = data_list

def buil_genre_data(file_list, config, event_genres):
    data_map = {}
    for file_name in file_list:
        sub_data = data_statistics(file_name, config)
        data_map[file_name] = sub_data

    statistics_results = {}
    genre_data = {}
    for event_genre in event_genres:
        statistics_results[event_genre] = 0

    count = 0
    for file_name in data_map.keys():
        datas = data_map[file_name]
        for data in datas:
            guid = data["guid"]
            genre = data["label"]
            classify_data(genre_data, genre, guid, event_genres)
            count = count + 1
    return genre_data

def split_genre_data(genre_data, event_genres):
    # 3. Count the total first and check whether it is about 69001
    total_data_count = sum(len(genre_data.get(t, [])) for t in event_genres)
    print("Total data count:", total_data_count)

    # 4. Calculate how many files are needed; hard-code if the count is confirmed
    max_per_file = 100
    num_files = (total_data_count // max_per_file) + (1 if total_data_count % max_per_file != 0 else 0)
    # If the target is confirmed as 691 files, it can also be set directly:
    # num_files = 691

    current_file_index = 0  # Used to distinguish output file names
    used_count = 0  # Number of assigned records

    sub_files = {}

    # 5. Start writing files one by one
    while used_count < total_data_count:
        file_data = []  # All records for the current file

        # Keep round-robin sampling across types until 100 records are collected or no records remain
        while len(file_data) < max_per_file:
            # One round: try to take one record from each type
            round_data = []
            for t in event_genres:
                # If this type still has records, take one
                if len(genre_data.get(t, [])) > 0:
                    guid = genre_data[t].pop(0)  # Take the first guid
                    round_data.append({"type": t, "guid": guid})

            # If this round takes nothing, all types are empty and sampling is complete
            if not round_data:
                break

            # Check whether this round fits in the current file
            space_left = max_per_file - len(file_data)
            if len(round_data) <= space_left:
                # Everything fits
                file_data.extend(round_data)
            else:
                # Only part of the round fits; fill up to 100
                file_data.extend(round_data[:space_left])
                # Put remaining records back into their original type for the next file
                leftovers = round_data[space_left:]
                for item in reversed(leftovers):
                    # Put them back at the head of the corresponding type to preserve order
                    genre_data[item["type"]].insert(0, item["guid"])
                # Current file is full, so exit
                break

        # If the current file received nothing, all data is exhausted
        if not file_data:
            break

        used_count += len(file_data)

        sub_files[current_file_index] = file_data
        # 6. Write the current file as JSON
        # filename = f"part_{current_file_index}.json"
        # with open(filename, 'w', encoding='utf-8') as f:
        #     json.dump(file_data, f, ensure_ascii=False, indent=2)
        #
        # print(f"Output file: {filename}, record count: {len(file_data)}")

        current_file_index += 1
    return sub_files
    # print("All files split; total assigned records:", used_count)

def dump_subfiles(subfiles_map, initial_datas, initial_index, config):
    file_datas = {}
    for file_index in subfiles_map.keys():
        subfiles = subfiles_map[file_index]
        subfile_datas = []
        for subfile in subfiles:
            target_guid = subfile["guid"]
            matching_item = next((item for item in initial_index if item["guid"] == target_guid), None)
            file_name = matching_item["file"]
            file_no = matching_item["file"].replace("initial_dataset_", "").replace(".json", "")
            # data_path = config.initial_dataset_root_path + file_name
            initial_data = initial_datas[file_no]
            data = next((item for item in initial_data if item["guid"] == target_guid), None)
            subfile_datas.append(data)
            print("test")
        file_datas[file_index] = subfile_datas
        filename = config.output_subfile_path + f"part_{file_index}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(subfile_datas, f, ensure_ascii=False, indent=2)
        print(str(file_index) + "is saved!")

def get_data(file_list, config):
    data_map = {}
    for file_name in file_list:
        output_root_path = config.initial_dataset_root_path + "initial_dataset_" + file_name + ".json"
        json_data = file_utils.read_json_file(output_root_path)
        data_map[file_name] = json_data
    return data_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=False, help="API key for accessing the service")
    parser.add_argument("--config", type=str, default="/tmp/echv_preprocessing/config/echv_config_split_files.json")

    # ************TEST************
    # args = parser.parse_args(["--api_key", "123"])
    args = parser.parse_args()
    # ************TEST************

    config = config.Config(args)
    output_subfile_path = config.output_subfile_path
    initial_index_path = config.initial_index_path
    initial_dataset_root_path = config.initial_dataset_root_path

    max_rows_per_file = config.max_rows_per_file
    event_genres = event_genres_without_sensdata.genre

    file_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    initial_index = file_utils.read_json_file(initial_index_path)
    genre_datas = buil_genre_data(file_list, config, event_genres)
    subfiles = split_genre_data(genre_datas, event_genres)
    initial_datas = get_data(file_list, config)
    results = dump_subfiles(subfiles, initial_datas, initial_index, config)
    print("finished!")
