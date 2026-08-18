import os
import json
import sys
import time
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import config.config_wikimedia_api as config
from utils import file_utils
from utils import preprocess_utils
from dataset.unbalanced_distribution.few_types import few_types
from dataset.unbalanced_distribution.few_types_count import few_types_count
from collections import defaultdict
from newspaper import Article
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import requests
from urllib.parse import quote

def search_wikinews(query, limit=50):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit
    }
    response = requests.get(wikinews_api, params=params)
    data = response.json()
    return data.get("query", {}).get("search", [])

def create_retry_session():
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "WangYao2025/1.0 (yaow2878@gmail.com)"
    })
    return session

def search_wikinews_paginated(query, total_limit=200, per_page=50):
    session = create_retry_session()  # ✅ 使用带 retry 的 session
    all_results = []
    offset = 0
    while len(all_results) < total_limit:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": per_page,
            "sroffset": offset
        }
        try:
            response = session.get(wikinews_api, params=params, timeout=10)
            data = response.json()
            page_results = data.get("query", {}).get("search", [])
            if not page_results:
                break
            all_results.extend(page_results)
        except Exception as e:
            print(f"🔴 Error during search_wikinews_paginated: {e}")
            break
        offset += per_page
        time.sleep(0.2)
    return all_results


def get_page_extract(page_title):
    session = create_retry_session()  # ✅ 使用带 retry 的 session
    safe_title = quote(page_title, safe='')  # ✅ URL 安全处理
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": page_title,
        "format": "json",
        "explaintext": 1
        # ,
        # "exintro": 1
    }
    try:
        response = session.get(wikinews_api, params=params, timeout=10)
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            return page.get("extract", "")
    except Exception as e:
        print(f"🔴 Error during get_page_extract for {page_title}: {e}")
    return ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/tmp/echv_preprocessing/config/echv_config_wikimedia_api.json")

    # ************TEST************
    args = parser.parse_args()
    # ************TEST************

    config = config.Config(args)
    initial_dataset_root_path = config.initial_dataset_root_path
    prompt_file_path = config.prompt_file_path
    hv_path = config.hv_path
    role1 = config.role1
    role2 = config.role2
    content1 = config.content1
    output_root_path = config.output_root_path + "hvs_addition_v7/"
    process_start = config.process_start
    process_end = config.process_end
    file_start = config.file_start
    file_end = config.file_end
    api_model = config.api_model
    max_rows_per_file = config.max_rows_per_file
    subfiles_root_path = config.subfiles_root_path
    min_thread_count = config.min_thread_count
    is_process = config.is_process
    completed_root_path = config.completed_output_root_path
    wikinews_api = config.wikinews_api
    wikinews_base_url = config.wikinews_base_url
    # output_file_name = config.output_file_name
    file_utils.check_and_create_file(output_root_path)
    hv_themes = file_utils.read_json_file("dataset/wikimedia/hv_theme_v7.json")

    for hv, value_keywords in hv_themes.items():
        if hv in ("Have freedom of thought", "Have freedom of action, Be loving"):
            continue
        # if hv == "Be loving":
        #     value_keywords = value_keywords[176:]
        for keyword in value_keywords:
            all_results = []
            output_file_path = output_root_path + hv + "/"
            file_utils.check_and_create_file(output_file_path)
            output_file = output_file_path + keyword + ".json"
            print(f"\n Searching for keyword: {keyword}")
            results = search_wikinews_paginated(keyword)
            for item in results:
                page_title = item['title']
                page_id = item['pageid']
                snippet = item.get("snippet", "")
                extract = get_page_extract(page_title)
                page_url = wikinews_base_url + quote(page_title.replace(" ", "_"))
                all_results.append({
                    "title": page_title,
                    "pageid": page_id,
                    "keyword": keyword,
                    "url": page_url,
                    "snippet_html": snippet,
                    "summary": extract
                })
                time.sleep(0.5)
            file_utils.save_json_data_list(output_file, all_results,2)
