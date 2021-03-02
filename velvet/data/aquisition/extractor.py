import lxml.html
import pandas as pd
import os

from bs4 import BeautifulSoup
from bs4 import SoupStrainer

# TODO: Repeated helper function
# TODO: Replace with iterator
def all_pages(just_nums = False):
    files = os.listdir('./pages/')

    if just_nums:
        files = [f.strip('.html') for f in files]

    return files

# TODO: Make parsing more dynamic
def parse():
    paths = all_pages()

    data = []

    for path in paths:
        tree = lxml.html.parse('./pages/'+path)
        root = tree.getroot()

        # TODO:// Make this clear that failed parsing is 
        # b/c page didn't return results
        try:
            title = root.find_class('job-show-title')[0].text_content()
            company = root.find_class('job-show-additional-item')[0].text_content()
            location = root.find_class('job-show-additional-item')[1].text_content()
            description = root.find_class('job-show-description')[0].text_content()

            data.append([title, company, location, description])
        except:
            pass

    df = pd.DataFrame(data, columns=['title','company', 'location','description'])
    df.to_csv('100k_jobs.csv')


# TODO: maybe make logging an option
# import cProfile
# cProfile.run('performance()', 'bs4_test.profile')

