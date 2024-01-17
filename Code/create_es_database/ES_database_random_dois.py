### LOAD DATA

from os import listdir
from os.path import isfile, join
from pathlib import Path
import pickle 
import os
from tqdm import tqdm
from datasets import Dataset

def get_filtered_abstracts(oa_folder,file,OPENALEXFOLDER):
    with open(oa_folder+file, 'rb') as f:
        data = pickle.load(f)
        f.close()
    doi = data['doi']
    abstract = data.get('pubmed', {}).get('abstract', None)
    year = data.get('year', 0)
    if year < 2013 or not abstract or len(abstract) < 100:
        return None
    if not os.path.exists(OPENALEXFOLDER+doi.replace('/', '_',1).lower()+'.txt'):
        # print('Not OPENALEX info')
        return None
    with open(OPENALEXFOLDER+doi.replace('/', '_',1).lower()+'.txt', 'rb') as f:
        OpenAlex = pickle.load(f)
        paper_concepts = set([concept['display_name'] for concept in OpenAlex['concepts']])
        f.close()
    target_concepts = set(['Medicine', 'Biology', 'Immunology', 'Cancer', 'Genetics'])

    # Check if there is an intersection between the paper's concepts and target concepts
    if not target_concepts.intersection(paper_concepts):
        return None
    return abstract

oa_folder = '/home/panchojasen/Data/OA_JSON/'

oa_folder2 = '/home/panchojasen/Data/abstracts_JSON/'
save_folder= '/home/panchojasen/Data/OA_vectors/'
OPENALEXFOLDER = '/home/panchojasen/Data/OpenAlex-info/'
import random

# Set the seed for reproducibility
random.seed(42)  # You can choose any number as the seed

# Assuming files_to_do is already defined as per your given code
# files_to_do = list(set([f for f in listdir(oa_folder) if isfile(join(oa_folder, f))]))

# Shuffle the list in place

# files_to_do = list(set([f for f in listdir(oa_folder) if isfile(join(oa_folder, f))]))
files_to_do = list(set([f for f in listdir(oa_folder) if isfile(join(oa_folder, f))]))

random.shuffle(files_to_do)
files_to_convert = []
counter = 0 
for file in tqdm(files_to_do):
    abstract = get_filtered_abstracts(oa_folder,file,OPENALEXFOLDER)
    if abstract and len(abstract) > 250:
        files_to_convert.append(file)
        counter += 1
    if counter > 60_000:
        break
print(len(files_to_convert))

with open('/home/panchojasen/Projects/NN-Engene/dois_to_vector/random_dois.pkl', 'wb') as file:
    pickle.dump(files_to_convert, file)