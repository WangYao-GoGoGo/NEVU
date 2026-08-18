"""Run preliminary zero-shot article genre classification for Phase-1 data."""

import os
import json
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import config.config as config
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from utils import file_utils
from dataset.wiki_ontology import event_genres
from utils import preprocess_utils
from transformers import pipeline


def truncate_content(content, max_tokens):
    """Truncate the content to fit within the model's input limit."""
    tokens = tokenizer.encode(content, truncation=True, max_length=max_tokens, add_special_tokens=False)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def tokenize_and_truncate(content, max_length, tokenizer):
    """Tokenize the content and truncate to a specified length using the provided tokenizer."""
    encoded = tokenizer(content, truncation=True, max_length=max_length, return_tensors="pt")
    truncated_content = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
    return truncated_content

classifier = pipeline("zero-shot-classification",
                      model="knowledgator/comprehend_it-base")

# Classification function
def classify_news(articles, types, classification_pipeline, output_article_genres_file):
    # classifications = []
    classifications_results = []

    count = 10
    for idx, article in enumerate(articles):
        start_time = time.time()
        article_content = article["content"]

        # Truncate article content to avoid exceeding the model input limit
        max_length = 512  # Limit to 512 tokens
        truncated_content = tokenize_and_truncate(article_content, max_length, classification_pipeline.tokenizer)

        if not truncated_content.strip():
            print(f"Warning: Article {idx} content is empty after tokenization. Skipping...")
            # classifications.append((article, "No valid content"))
            continue

        # sequence_to_classify = "one day I will see the world"
        # candidate_labels = ['travel', 'cooking', 'dancing']
        result = classifier(truncated_content, types)
        # top_categories = list(zip(result["labels"][:3], result["scores"][:3]))  # Get the top-scoring results

        # Sort by score and get the highest-scoring results
        sorted_results = sorted(zip(result["labels"], result["scores"]), key=lambda x: x[1], reverse=True)
        top_categories = sorted_results[:1]  # Get the top-scoring results

        # classifications.append((article, top_categories))

        # Extract labels and scores
        # labels = result['labels'][:3]
        # scores = result['scores'][:3]

        # Print results
        # for label, score in top_categories:

        #     print(f"Label: {label}, Score: {score}")
        article_guid = article["guid"]
        label, score = top_categories[0]

        completed_guid = "{\"guid\": \"" + article_guid + "\", \"label\": \"" + label  + "\", \"score\": \"" + score + "\"}"
        classifications_results.append(completed_guid)

        end_time = time.time()
        hours, minutes, seconds = preprocess_utils.spent_time(start_time, end_time)
        log_con = f"*******current file name: {file_name}, model_name: {model_name}, current num: {idx}, SpentTimeforPerArticle: {hours}h {minutes}m {seconds:.2f}s ********"
        print(log_con)

        if (idx + 1) % count == 0 or idx == (len(articles) - 1 ):
            json_article_genres = json.loads(str(completed_guid))
            file_utils.append_json_data_to_file(output_article_genres_file, json_article_genres, 4)
            classifications_results = []
            print(str(idx) + "is saved!")


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
