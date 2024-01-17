from sentence_transformers import SentenceTransformer, util
import time
import pickle
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import torch
from sentence_transformers import InputExample


oa_folder = '/home/panchojasen/Data/OA_JSON/'
def get_abstract_doi_paper(oa_folder,file):
    # s = time.time()
    with open(oa_folder+file, 'rb') as f:
        data = pickle.load(f)
        f.close() 
    try:
        abstract = data['pubmed']['abstract']
        doi = data['doi']
        return doi, abstract
    except:
         return None, None
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pickle
import os
import torch
model_name = "google/pegasus-xsum"

device = "cuda" if torch.cuda.is_available() else "cpu"
save_folder = "/home/panchojasen/Data/query_pegasus/"

tokenizer = PegasusTokenizer.from_pretrained(model_name)

model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

def summary_abstract(dois, abstracts, save_folder):
    batch = tokenizer(abstracts, truncation=True, padding="longest", return_tensors="pt").to(device)

    translated = model.generate(**batch)

    queries = tokenizer.batch_decode(translated, skip_special_tokens=True)
    create_and_save_dicts(dois, queries, save_folder)

def create_and_save_dicts(dois, queries,save_folder):
    """
    Create a dictionary from two lists: 'dois' and 'queries', 
    and save each key-value pair as a separate pickle file.

    :param dois: List of DOIs (keys)
    :param queries: List of queries (values)
    :return: None
    """
    # Create the dictionary
    doi_query_dict = dict(zip(dois, queries))

    # Check and create a directory for storing pickle files

    # Write each key-value pair to a separate pickle file
    for doi, query in doi_query_dict.items():
        file_path = os.path.join(save_folder, doi.replace('/', '_') + ".pkl")
        with open(file_path, 'wb') as file:
            pickle.dump({doi: query}, file)

    # print(f"Saved {len(doi_query_dict)} files in '{save_folder}' directory.")


    

dois = []
abstracts = []
counter = 0
batch_size = 4
lens = 0
counter = 0
for file in tqdm(reversed(listdir(oa_folder))):
    counter += 1
    if counter < 10_000:
        continue
    if not isfile(join(oa_folder, file)):
        print('Not a file')
        continue
    # if len(abstracts) > 250_000:
    #     break

    doi, abstract = get_abstract_doi_paper(oa_folder, file)
    # print(len(abstract))
    if doi is not None and abstract is not None:
        lens += len(abstract)
        dois.append(doi)
        abstracts.append(abstract)
        counter += 1
        if counter == batch_size or lens > 6000:
            summary_abstract(dois,abstracts, save_folder)
            dois = []
            abstracts = []
            # Reset the counter
            counter = 0
            lens = 0
print('END')










