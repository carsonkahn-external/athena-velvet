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


def build_urls():
    base = 'http://www.velvetjobs.com/job-posting/fixed-income-distressed-research-analyst-vice-president-'

    bottom = 200000
    count = 100000

    #and not in set
    already_scraped = all_pages(just_nums = True)
    urls = [base+str(x) for x in range(bottom, bottom+count) if str(x) not in already_scraped]

    

    clean_urls = [] #urls

    # #TODO: fix
    # for url in urls:
    #     is_in = False
    #     for num in already_scraped:
    #         if num == url.split('-')[-1]:
    #             is_in = True
    #     if not is_in:
    #         clean_urls.append(url)

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

    test_url = 'http://ip.smartproxy.com/json'
    li_url = 'https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search&trkf_C=1382'

    response = requests.get(url, headers=generate_header(), proxies=proxies)
    # print(response.json())

    time.sleep(random.uniform(.1, 1.0))

    return response

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

def run_one():
    urls = build_urls()

    response = fetch_url(urls[0])

    page_number = response.url.split('-')[-1]
    with open(f'./pages/{page_number}.html', 'w+') as fh:
        fh.write(response.text)


def parse_velvet():
    paths = all_pages()
    data = []

    for path in paths:
        with open('./pages/'+path, 'r') as fh:
            soup = BeautifulSoup(fh, 'html.parser')
        try:
            title = soup.select('.job-show-title')[0].get_text()
            location = soup.select('.job-show-additional-item')[0].get_text()
            description = soup.select('.job-show-description')[0].get_text()

            data.append([title, location, description])
        except:
            pass

    df = pd.DataFrame(data, columns=['title','location','description'])

    df.to_csv('job_postings.csv')

run()