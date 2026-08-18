import file_utils as file_utils

def compute_start_index(search_guid, file_path):
    """Returns the number of tokens in a text string."""
    datas = file_utils.read_json_file(file_path)
    for idx, data in enumerate(datas):
        guid = data["guid"]
        if guid == search_guid:
            return idx
    return null

if __name__ == '__main__':
    file_path = "dataset/initial_corpus/initial_dataset_1.json"
    start_index = compute_start_index("0-4327-1-t", file_path)
    print(start_index)