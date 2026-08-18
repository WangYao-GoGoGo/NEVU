"""Classify collected news articles into event genres for Phase-1 filtering."""

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
def classify_news(articles, types, classification_pipeline, output_article_genres_file, max_length, max_save_count, process_start):
    # classifications = []
    classifications_results = []

    # count = 10
    for idx, article in enumerate(articles):
        start_time = time.time()
        article_content = article["content"]

        # Truncate article content to avoid exceeding the model input limit
        # max_length = 512  # Limit to 512 tokens
        truncated_content = tokenize_and_truncate(article_content, max_length, classification_pipeline.tokenizer)

        if not truncated_content.strip():
            print(f"Warning: Article {idx + process_start} content is empty after tokenization. Skipping...")
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
        article_guid = article["guid"].replace("\n", "\\n")
        label, score = top_categories[0]

        completed_guid = "{\"guid\": \"" + article_guid + "\", \"label\": \"" + label  + "\", \"score\": \"" + str(score) + "\"}"
        classifications_results.append(json.loads(completed_guid))

        end_time = time.time()
        hours, minutes, seconds = preprocess_utils.spent_time(start_time, end_time)
        log_con = f"*******current file name: {file_name}, model_name: {model_name}, current num: {idx + process_start}, SpentTimeforPerArticle: {hours}h {minutes}m {seconds:.2f}s ********"
        print(log_con)

        if (idx + 1) % max_save_count == 0 or idx == (len(articles) - 1 ):
            # json_article_genres = json.loads(classifications_results)
            file_utils.save_json_data_list(output_article_genres_file, classifications_results, 4)
            classifications_results = []
            print(str(idx + process_start) + " is saved!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=False, help="API key for accessing the service")
    parser.add_argument("--config", type=str, default="/tmp/echv_preprocessing/config/echv_config.json")

    # ************TEST************
    # args = parser.parse_args(["--api_key", "123"])
    args = parser.parse_args()
    # ************TEST************

    config = config.Config(args)
    file_name = config.file_name
    initial_dataset_root_path = config.initial_dataset_root_path
    prompt_file_path = config.prompt_file_path
    hv_path = config.hv_path
    initial_dataset_file_path = initial_dataset_root_path + file_name  + ".json"
    datas = file_utils.read_json_file(initial_dataset_file_path)
    # datas = datas.to("cuda")
    model_name = config.model_name
    process_start = config.process_start
    datas = datas[process_start:]

    output_root_path = config.output_root_path  + file_name
    file_utils.check_and_create_file(output_root_path)
    output_article_genres_file = output_root_path + "/article_genres.json"

    max_length = config.max_length
    max_save_count = config.max_save_count


    # Initialize the LLAMA2 model and tokenizer
    # model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with an appropriate model name

    # tokenizer_path = "/home/iiserver33/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/"
    # model_path = "/home/iiserver33/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/"

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
    cla_pipeline = pipeline("zero-shot-classification", model="knowledgator/comprehend_it-base")

    article_genres = event_genres.genre

    # Classify news articles
    classify_news(datas, article_genres, cla_pipeline, output_article_genres_file, max_length, max_save_count, process_start)

    # Output classification results
    # for article, category in results:
    #     print(f"News: {article}\nPredicted Category: {category}\n")
