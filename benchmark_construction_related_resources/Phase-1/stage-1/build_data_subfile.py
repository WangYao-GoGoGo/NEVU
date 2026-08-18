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
    # 3. 先统计总数，看看是不是约为 69001
    total_data_count = sum(len(genre_data.get(t, [])) for t in event_genres)
    print("总数据量:", total_data_count)

    # 4. 计算需要多少文件（如果你确认是 691 个，可直接写死）
    max_per_file = 100
    num_files = (total_data_count // max_per_file) + (1 if total_data_count % max_per_file != 0 else 0)
    # 若你确定要 691 个文件，也可以直接：
    # num_files = 691

    current_file_index = 0  # 用来命名文件时做区分
    used_count = 0  # 已分配的数据条数

    sub_files = {}

    # 5. 开始逐文件写入
    while used_count < total_data_count:
        file_data = []  # 当前文件的所有条目

        # 不断做“轮询各类型”的操作，直到凑满 100 条或取不出了
        while len(file_data) < max_per_file:
            # 做一轮：尝试给每个类型都取一条
            round_data = []
            for t in event_genres:
                # 若该类型还有剩余数据，就取出1条
                if len(genre_data.get(t, [])) > 0:
                    guid = genre_data[t].pop(0)  # 取出第一个guid
                    round_data.append({"type": t, "guid": guid})

            # 如果这一轮什么也没取到，说明所有类型都空了 => 全部取完
            if not round_data:
                break

            # 检查本轮能否全部放进当前文件
            space_left = max_per_file - len(file_data)
            if len(round_data) <= space_left:
                # 全部放得下
                file_data.extend(round_data)
            else:
                # 只能放一部分，凑满到100
                file_data.extend(round_data[:space_left])
                # 剩余的条目还需要“放回原类型”里去，以便下一个文件继续用
                leftovers = round_data[space_left:]
                for item in reversed(leftovers):
                    # 放回到对应类型的头部，保证下次取的时候顺序不乱
                    genre_data[item["type"]].insert(0, item["guid"])
                # 当前文件已经满了，就退出
                break

        # 如果当前文件什么都没取到，说明数据已用尽，结束
        if not file_data:
            break

        used_count += len(file_data)

        sub_files[current_file_index] = file_data
        # 6. 写入当前文件 (json 格式示例)
        # filename = f"part_{current_file_index}.json"
        # with open(filename, 'w', encoding='utf-8') as f:
        #     json.dump(file_data, f, ensure_ascii=False, indent=2)
        #
        # print(f"输出文件: {filename}, 条目数: {len(file_data)}")

        current_file_index += 1
    return sub_files
    # print("全部文件分割完成，共分配数据条数:", used_count)

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