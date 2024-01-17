import os
from pubmed_api.pymed.api import PubMed
# from pymed.api import PubMed

# from pymed.api import PubMed
from datetime import timedelta
from datetime import datetime
import requests
from pubmed_api.constants.JOURNALS import JOURNALS
# from constants.JOURNALS import JOURNALS

import time

class PubMedAPI:  
    pubmed = PubMed(tool="MyTool", email="adrian_3091@hotmail.com")
    my_api_key = '8ae84e14750ca15c7680a399302a5b0ecf08'
    pubmed.parameters.update({'api_key': my_api_key})
    _query = ' OR '.join([f'{journal}[Journal]' for journal in JOURNALS])

    def __init__(self):
        pass
        # self._query += f' AND ("{self.older_date}"[Date - Create] : "{self.newer_date}"[Date - Create])'

    @staticmethod
    def get_last_scraped_date():
        # Check in the folder ../../registry if there is a file called pubmed_last_scraped_date.txt with a date in the
        # format YYYY/MM/DD and return it, and if it doesn't exist, return the date 1900/01/01
        with open(os.path.join('registry', 'pubmed_last_scraped_date.txt'), 'a+') as f:
            try:
                last_scraped_date = f.read()
                if not last_scraped_date:
                    last_scraped_date = '1900/01/01'
            except FileNotFoundError:
                last_scraped_date = '1900/01/01'

        return datetime.strptime(last_scraped_date, '%Y/%m/%d')

    @classmethod
    def query(cls, older_date=None, newer_date=None, max_results=-1):
        if not older_date:
            older_date = cls.get_last_scraped_date()
        older_date = datetime.strptime(older_date, '%Y/%m/%d')
        if  newer_date:
            newer_date = datetime.strptime(newer_date, '%Y/%m/%d')
        else: 
            newer_date = datetime.now()

        # We go one day by one just to make sure we don't surpass the limit of 'retstart' < 10000 of the PubMed API
        delta = newer_date - older_date
        for i in reversed(range(delta.days + 1)):
            _day = older_date + timedelta(days=i)
            _query = f'{cls._query} AND ("{_day}"[Date - Create] : "{_day}"[Date - Create])'
            max_retries = 4
            backoff_factor = 4

            for retry in range(max_retries):
                try:
                    # I'm assuming `cls.pubmed.query` is the method that might raise the ConnectionError
                    yield from cls.pubmed.query(_query, max_results)
                    break  # If the query succeeds, we break out of the loop
                except requests.exceptions.ConnectionError as e:
                    if retry == max_retries - 1:  # If this was the last retry
                        print(f"Connection error occurred: {e}. Skipping")
                        raise  # Re-raise the last exception
                    else:
                        print(f"Connection error occurred: {e}. Attempting again....")
                        time.sleep(backoff_factor * (2 ** retry))  # Exponential backoff
                except requests.exceptions.RequestException as e: 
                    # You can catch other exceptions here as well
                    print(f"An error occurred: {e}")
                    break
    @classmethod
    def query_doi(cls,doi,max_results=10):        
        if isinstance(doi,list):
            max_results=1000
            _query = ' OR '.join(doi)
            # print(_query)
        else:
            _query = doi
        max_retries = 4
        backoff_factor = 4
        for retry in range(max_retries):
            try:
                # I'm assuming `cls.pubmed.query` is the method that might raise the ConnectionError
                yield from cls.pubmed.query(_query, max_results)
                break  # If the query succeeds, we break out of the loop
            except requests.exceptions.ConnectionError as e:
                if retry == max_retries - 1:  # If this was the last retry
                    print(f"Connection error occurred: {e}. Skipping")
                    raise  # Re-raise the last exception
                else:
                    print(f"Connection error occurred: {e}. Attempting again....")
                    time.sleep(backoff_factor * (2 ** retry))  # Exponential backoff
            except requests.exceptions.RequestException as e: 
                # You can catch other exceptions here as well
                print(f"An error occurred: {e}")
                break   
            
            
            
if __name__ == '__main__':
    pubmed_api = PubMedAPI()
    # hit= a.query_doi('10.1007/s10147-013-0635-5')
    for hit in pubmed_api.query_doi('10.1007/s10147-013-0635-5'):
        print(hit.publication_types)
    results = pubmed_api.query(older_date=older_date,newer_date = newer_date )
    dois = {}
    alreadyin=0
    for article in results:
        if article.doi == None:
            continue