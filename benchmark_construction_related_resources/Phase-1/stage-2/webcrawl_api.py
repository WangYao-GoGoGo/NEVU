"""Reference web-crawling utilities for collecting supplementary news articles."""

import os
import json
import sys
import time
import threading
import feedparser
from bs4 import BeautifulSoup
from dataset.wikimedia import social_media_2 as sm


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import config.config_webcrawl_api as config
from utils import file_utils
from utils import preprocess_utils
from utils import token_utils
from dataset.unbalanced_distribution.few_types import few_types
from dataset.unbalanced_distribution.few_types_count import few_types_count
from collections import defaultdict
from newspaper import Article
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from collections import Counter

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
#
# service = Service(ChromeDriverManager().install())
# driver = webdriver.Chrome(service=service, options=options)
# import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

import requests
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


def get_news_from_newsapi():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "education",
        "sources": "bbc-news",
        "apiKey": "YOUR_NEWSAPI_KEY",
        "pageSize": 100,
        "page": 1
    }
    res = requests.get(url, params=params)
    articles = res.json()["articles"]
    return articles

def get_bbc_education_articles(pages=5):
    all_links = []
    base_url = "https://www.bbc.com/news/education"
    for i in range(pages):
        url = base_url if i == 0 else f"{base_url}?page={i+1}"
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        for a in soup.select("a.gs-c-promo-heading"):
            href = a.get("href")
            if href and href.startswith("/news/education"):
                full_url = "https://www.bbc.com" + href
                all_links.append(full_url)
    return list(set(all_links))

def get_news_from_newsdata(domain, qs, apikey):
    url = "https://newsdata.io/api/1/news"
    all_articles = []
    for q in qs:
        params = {
            "apikey": apikey,
            "q": q,
            "domain": domain,
            "language": "en",
            "page": 1,
            "page_size": 100
        }
        res = requests.get(url, params=params)
        if res.status_code == 200:
            articles = res.json().get("results", [])
            all_articles.extend(articles)
        else:
            print(f"Failed request for q='{q}', domain='{domain}': {res.status_code}")
    return all_articles

def get_articles_from_rss(rss_url):
    feed = feedparser.parse(rss_url)
    # urls = [entry.link for entry in feed.entries]
    return feed

def get_articles_from_html(base_url, selector, limit=100, headers=None):
    print(f"Visiting: {base_url}")
    headers = headers or {"User-Agent": "xxx"}
    try:
        res = requests.get(base_url, headers=headers, timeout=10)
        if res.status_code != 200:
            print(f"Failed to fetch {base_url}: {res.status_code}")
            return []
        soup = BeautifulSoup(res.text, "html.parser")
        links = soup.select(selector)
        urls = []
        for a in links:
            href = a.get("href")
            if href and not href.startswith("http"):
                href = "https://www.bbc.com" + href
            if href not in urls:
                urls.append(href)
            if len(urls) >= limit:
                break
        print(f"Fetched {len(urls)} URLs from {base_url}")
        return urls
    except Exception as e:
        print(f"Error fetching {base_url}: {e}")
        return []

def download_articles(feed):
    articles = []
    for entry in feed.entries:
        url = entry.link
        published = entry.published
        if url:
            guid = entry.id
            try:
                article = Article(url)
                article.download()
                article.parse()
                articles.append({
                    "guid": guid,
                    "title": article.title,
                    "url": url,
                    "content": article.text,
                    "published": published,
                })
                time.sleep(1)
            except Exception as e:
                print(f"[Error] {url}: {e}")
    return articles

def get_articles_from_html(base_url, css_selector, limit):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    print(f"Opening: {base_url}")
    driver.get(base_url)
    time.sleep(5)  # Wait for content to load

    # Scroll the page to trigger lazy loading
    scroll_pause = 2
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height or len(driver.find_elements(By.CSS_SELECTOR, css_selector)) >= limit:
            break
        last_height = new_height

    elements = driver.find_elements(By.CSS_SELECTOR, css_selector)
    links = []
    for el in elements:
        href = el.get_attribute("href")
        if href and href not in links:
            links.append(href)
        if len(links) >= limit:
            break

    driver.quit()
    print(f"Fetched {len(links)} links from {base_url}")
    return links

def statistics(folder_path, global_name):
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    total_data = []
    for file_name in file_names:
        file_path = folder_path + file_name
        output_files = file_utils.read_json_file(file_path)
        if global_name == "pageid":
            for output_file in output_files:
                if "pageid" in output_file:
                    output_file["guid"] = output_file.pop("pageid")
                    output_file["published"] = output_file.pop("date")

        for output_file in output_files:
            check_result = check_exist(output_file["guid"], total_data)
            if not check_result:
                content = output_file["content"]
                input_count = token_utils.num_tokens_from_string(content, "cl100k_base")
                print("********Input token count: " + str(input_count))
                if input_count > 200:
                    total_data.append(output_file)
    return total_data

def check_exist(key_guid, datas):
    for data in datas:
        guid = data["guid"]
        if key_guid == guid:
            return True
    return False

def filter_duplicated(results, results_5):
    total_data = []
    total_data.extend(results_5)
    for result in results:
        check_result = check_exist(result["guid"], total_data)
        if not check_result:
            total_data.append(result)
    return total_data

def split_data(total_results, max_limit):
    subfiles = {}
    current_page = []
    for idx, total_result in enumerate(total_results):
        current_page.append(total_result)
        if len(current_page) == max_limit:
            curr_page_num = idx // max_limit
            subfiles[curr_page_num] = current_page
            current_page = []

    # Add the final page with fewer than max_limit records
    if current_page:
        curr_page_num = len(subfiles)
        subfiles[curr_page_num] = current_page

    return subfiles

def dump_subfiles(dump_file_path, dump_subfile_name, subfiles, start_file):
    for filename, subfile in subfiles.items():
        temp_file_path = dump_file_path + str(filename + start_file) + "/"
        file_utils.check_and_create_file(temp_file_path)
        temp_file = temp_file_path + dump_subfile_name
        file_utils.dump_json_file(temp_file, subfile, 4)

def statistic_and_dump_splited_files(output_file_path_4, output_file_path_5, output_file_path_6):
    results_4 = statistics(output_file_path_4, "guid")
    results_5 = statistics(output_file_path_5, "guid")
    results_6 = statistics(output_file_path_6, "pageid")
    # tmp_results = filter_duplicated(results_4, results_5)
    # total_results = filter_duplicated(tmp_results, results_6)
    total_results = filter_duplicated(results_6, results_6)
    subfiles = split_data(total_results, max_rows_per_file)
    # Set this to 12 because 12 files had already been created; the last file was 11, so the second batch starts at file 12.
    dump_subfiles(dump_subfile_path, dump_subfile_name, subfiles, 12)
    print(results)

def data_crawling(output_dir):
    file_utils.check_and_create_file(output_dir)
    #
    # # Step 1: Extract from RSS
    for name, rss_url in rss_feeds.items():
        print(f"Fetching from RSS: {name}")
        feed = get_articles_from_rss(rss_url)
        articles = download_articles(feed)
        # save_json_data_list(os.path.join(output_dir, f"{name}.json"), articles)
        file_utils.save_json_data_list(os.path.join(output_dir, f"{name}.json"), articles, 2)

    # Step 2: Supplement from HTML across multiple pages
    for name, conf in html_sources.items():
        print(f"Fetching from HTML: {name}")
        # urls = get_articles_from_html(conf["base_url"], conf["selector"], limit=1500)
        if name == "BBC_Travel":
            links = ['https://www.bbc.com/travel/article/20250710-a-vintage-ride-on-the-british-isles-only-electric-mountain-railway', 'https://www.bbc.com/travel/article/20250710-an-insiders-summer-guide-to-italy', 'https://www.bbc.com/travel/article/20250115-the-25-best-places-to-travel-in-2025', 'https://www.bbc.com/travel/article/20250708-the-best-places-to-eat-pizza-in-naples', 'https://www.bbc.com/travel/article/20250707-inside-italys-secret-mosaic-school', 'https://www.bbc.com/travel/article/20250211-giada-de-laurentiis-family-guide-to-rome', 'https://www.bbc.com/travel/article/20250703-the-ski-resort-olympians-flock-to-to-each-summer', 'https://www.bbc.com/travel/article/20250703-where-to-go-instead-of-the-big-us-parks-this-summer', 'https://www.bbc.com/travel/article/20250625-the-beach-town-that-became-a-bitcoin-testbed', 'https://www.bbc.com/travel/article/20250630-americans-react-to-the-us-worldwide-caution-alert', 'https://www.bbc.com/travel/article/20250626-how-school-stays-are-saving-japans-dying-villages', 'https://www.bbc.com/travel/article/20250625-four-countries-welcoming-sports-fans', 'https://www.bbc.com/travel/article/20250620-a-technicolour-visit-to-moroccos-holiest-town', 'https://www.bbc.com/travel/article/20250626-how-chiwetel-ejiofor-is-spending-his-summer-in-london', 'https://www.bbc.com/travel/article/20250623-the-archdruid-of-stonehenges-guide-to-glastonbury', 'https://www.bbc.com/travel/article/20250602-jimmy-choos-guide-to-kuala-lumpur', 'https://www.bbc.com/travel/article/20250527-atsuko-okatsukas-guide-to-los-angeles', 'https://www.bbc.com/travel/article/20250514-an-indycar-drivers-guide-to-indianapolis', 'https://www.bbc.com/travel/article/20250513-the-pasta-queens-favourite-cacio-e-pepe-in-rome', 'https://www.bbc.com/travel/article/20240517-a-weekend-in-cannes-with-an-attach-to-the-stars', 'https://www.bbc.com/travel/article/20250505-where-to-get-new-york-citys-best-chinese-food', 'https://www.bbc.com/travel/article/20250618-why-oslo-might-be-europes-most-liveable-city-break', 'https://www.bbc.com/travel/article/20250516-monacos-new-neighbourhood-rising-out-of-the-sea', 'https://www.bbc.com/travel/article/20250424-the-indian-oceans-laid-back-paradise-on-earth', 'https://www.bbc.com/travel/article/20250319-the-cook-islands-paradise-that-doesnt-want-to-be-hawaii', 'https://www.bbc.com/travel/article/20250704-why-italys-national-parks-are-perfect-for-foodies', 'https://www.bbc.com/travel/article/20250702-a-journey-through-the-united-states-of-barbecue', 'https://www.bbc.com/travel/article/20250627-the-european-nation-pioneering-beer-diplomacy', 'https://www.bbc.com/travel/article/20250625-indias-cooling-summer-dish-that-costs-less-than-a-dollar', 'https://www.bbc.com/travel/article/20250710-italys-sunken-city-returning-from-the-sea', 'https://www.bbc.com/travel/article/20250702-barga-the-most-scottish-town-in-italy', 'https://www.bbc.com/travel/article/20250704-six-new-and-upcoming-summer-travel-books-that-inspire-wonder', 'https://www.bbc.com/travel/article/20250702-the-unlikely-2025-uk-city-of-culture', 'https://www.bbc.com/travel/article/20250627-the-big-change-affecting-european-travel', 'https://www.bbc.com/travel/article/20250623-attabad-lake-pakistan-the-stunning-legacy-of-a-natural-disaster', 'https://www.bbc.com/travel/article/20250620-paddling-the-dramatic-grand-canyon-of-canada', 'https://www.bbc.com/travel/article/20250618-drive-your-own-tuk-tuk-in-sri-lanka', 'https://www.bbc.com/travel/article/20250618-the-bangkok-death-cafe-that-changed-my-life']
        else:
            links = get_articles_from_html(conf["base_url"], conf["selector"], limit=1500)
        if links:
            # feed_2 = feedparser.parse(link)
            # articles = download_articles(feed_2)
            all_results = []
            for link in links:
                article = Article(link)
                article.download()
                article.parse()
                article.images
                all_results.append({
                    "title": article.title,
                    "pageid": article.url,
                    "date": article.publish_date.isoformat() if article.publish_date else None,
                    "keyword": name,
                    "url": link,
                    "images": list(article.images) if article.images else [],
                    "content": article.text
                })
                time.sleep(1)  # Avoid overly frequent requests
            file_utils.save_json_data_list(os.path.join(output_dir, f"{name}.json"), all_results, 2)

    # for domain, rss_url in rss_feeds.items():
    #     print("========current:=========" + domain)
    #     feed = feedparser.parse(rss_url)
    #     output_file_path_domain = output_file_path + domain + "/"
    #     file_utils.check_and_create_file(output_file_path_domain)
    #     output_file = output_file_path_domain + "output.json"
    #     all_results = []
    #     for entry in feed.entries:
    #         url = entry.link
    #         page_id = entry.id
    #         date = entry.published
    #         article = Article(url)
    #         article.download()
    #         article.parse()
    #         all_results.append({
    #             "title": article.title,
    #             "pageid": page_id,
    #             "date": date,
    #             "keyword": domain,
    #             "url": url,
    #             "content": article.text
    #         })
    #     time.sleep(1)  # Avoid overly frequent requests
    #     file_utils.save_json_data_list(output_file, all_results, 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/tmp/echv_preprocessing/config/echv_config_webcrawl_api.json")

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
    output_file_path_4 = config.output_root_path + "personal_focus4/"
    output_file_path_5 = config.output_root_path + "personal_focus5/"
    output_file_path_6 = config.output_root_path + "personal_focus6/"
    dump_subfile_path = config.dump_subfile_path
    dump_subfile_name = config.dump_subfile_name
    # hv_themes = file_utils.read_json_file("dataset/wikimedia/hv_theme_v7.json")
    # apikey = config.newsdata_api_key
    # charge of fees!!!
    # domains = ["bbc.com/worklife", "bbc.com/education", "bbc.com/future", "bbc.com/reel", "learningenglish.voanews.com/z/3612", "voanews.com/z/6248	"]
    # for domain in domains:
    #     results = get_news_from_newsapi(domain, qs)
    #     print("finished!")

    # value_counts = Counter(rss_feeds.values())
    # # # Keep only entries whose value appears once
    # filtered_rss_feeds = {k: v for k, v in rss_feeds.items() if value_counts[v] == 1}

    html_sources = sm.html_sources
    # #
    # print(sm.__file__)
    rss_feeds = sm.rss_feeds
    # output_dir = output_file_path
    output_dir_6 = output_file_path_6
    # data_crawling(output_dir_6)


    # Deduplicate and batch crawled data into subfiles for Phase-1 processing

    statistic_and_dump_splited_files(output_file_path_4, output_file_path_5, output_file_path_6)

