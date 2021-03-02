import json
import os
import pickle
import random
import time

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd

from fake_useragent import UserAgent
from bs4 import BeautifulSoup



def all_pages(just_nums = False):
    files = os.listdir('./pages/')

    if just_nums:
        files = [f.strip('.html') for f in files]

    return files

# TODO: make base configurable
def build_urls():
    base = 'http://www.velvetjobs.com/job-posting/fixed-income-distressed-research-analyst-vice-president-'

    bottom = 200000
    count = 100000

    already_scraped = all_pages(just_nums = True)
    urls = [base+str(x) for x in range(bottom, bottom+count) 
                if str(x) not in already_scraped]

    random.shuffle(urls)

    return urls


def fetch_url(url):
    ua = UserAgent()

    def generate_header():
        headers = {
            'User-Agent': ua.random,
            'DNT': '1',
            'Connection': 'close'
        }

        return headers

    proxies = {
      'http':'http://user-spa634357b:@r@chnid@gate.smartproxy.com:7000'
      #'https': 'https://user-spa634357b:@r@chnid@gate.smartproxy.com:7000'
    }

    # TODO: Break out proxy testing
    test_url = 'http://ip.smartproxy.com/json'
    li_url = 'https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search&trkf_C=1382'

    response = requests.get(url, headers=generate_header(), proxies=proxies)
    # print(response.json())

    time.sleep(random.uniform(.1, 1.0))

    return response

# TODO: Better error handling
# probably better to use queue
def run():
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        urls = build_urls()

        url_futures = (executor.submit(fetch_url, url) for url in urls)

        for future in concurrent.futures.as_completed(url_futures):
                try:
                    response = future.result()

                except Exception as exc:
                    response = str(type(exc))
                finally:
                    try:
                        page_number = response.url.split('-')[-1]
                        with open(f'./pages/{page_number}.html', 'w+') as fh:
                            fh.write(response.text)
                    except:
                        pass

# TODO: Make running amount amount configurable
def run_one():
    urls = build_urls()

    response = fetch_url(urls[0])

    page_number = response.url.split('-')[-1]
    with open(f'./pages/{page_number}.html', 'w+') as fh:
        fh.write(response.text)

