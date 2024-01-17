import pandas as pd
import os
os.chdir('/home/panchojasen/Projects/NN-Engene/lib')
from Custom_model_Evaluation import Custom_model_Evaluation

# df = pd.read_csv('../dois_query.csv')

df = pd.read_csv('/home/panchojasen/Projects/NN-Engene/dois_query_text.csv', encoding='latin1', sep = ';')

df.columns = ['dois', 'query', 'exclude']
df['dois'] = df['dois'].apply(lambda x: x.replace(';;', '').strip())
df['dois'] = df['dois'].apply(lambda x: x.replace(';', '').strip())
df['query'] = df['query'].apply(lambda x: x.strip())
df['query'] = df['query'].apply(lambda x: x.lower())

df = df[df['exclude'] == 0]

query_to_doi_dict = df.groupby('query')['dois'].apply(list).to_dict()


from elasticsearch_dsl import Document, DenseVector, Text, Keyword, Index, connections, Nested
from elasticsearch_dsl.query import MatchAll
from os import listdir
from os.path import isfile, join
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

models = []
# models.append({'msmarco-distilbert-cos-v5': 'sentence-transformers/msmarco-distilbert-cos-v5'})
# models.append({'test_model2':'/home/panchojasen/Projects/PaperScraper/Sentence-transformers/test_model2'})
# models.append({'Model_MSMARCO_pegasus':'/home/panchojasen/Projects/NN-Engene/Code/Model_MSMARCO_pegasus'})
# models.append({'msmarco-distilbert-cos-v5-finetuned-test-delate':'/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/msmarco-distilbert-cos-v5-finetuned-test-delate'})

models.append({'Model_MSMARCO_pegasus':'/home/panchojasen/Projects/NN-Engene/Code/Model_MSMARCO_pegasus'})
models.append({'S-PubMedBert-MS-MARCO': 'pritamdeka/S-PubMedBert-MS-MARCO'})
models.append({'Model_Pubmed_pegasus_allqueries':'/home/panchojasen/Projects/NN-Engene/Code/Model_Pubmed_pegasus_allqueries'})





# models.append({'Model_Pubmed_pegasus_allqueries':'/home/panchojasen/Projects/NN-Engene/Code/Model_Pubmed_pegasus_allqueries'})
# models.append({'Model_MSMARCO_pegasus':'/home/panchojasen/Projects/NN-Engene/Code/Model_MSMARCO_pegasus'})
# models.append({'test_model_all':'/home/panchojasen/Projects/PaperScraper/Sentence-transformers/test_model_all'})
# models.append({'test_model2':'/home/panchojasen/Projects/PaperScraper/Sentence-transformers/test_model2'})

# models.append({'msmarco-distilbert-cos-v5-finetuned-test-delate':'/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/msmarco-distilbert-cos-v5-finetuned-test-delate'})
# models.append({'Model_pubmed_abstracts':'/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/Model_pubmed_abstracts'})

# models.append({'S-PubMedBert-MS-MARCO': 'pritamdeka/S-PubMedBert-MS-MARCO'})

results = {}
for model in models:
    index_name, model_name = next(iter(model.items()))
    results.update(Custom_model_Evaluation(model_name,query_to_doi_dict, index_name.lower(), k=5))
results



metrics = list(results[list(results.keys())[0]].keys())

# import seaborn as sns

# Function to shorten model names
def shorten_model_names(name):
    parts = name.split('_')
    if len(parts) > 3:
        return '_'.join(parts[:3]) + '...'
    return name

# Shortened model names for x-axis labels
shortened_model_names = [shorten_model_names(name) for name in results.keys()]
shortened_model_names = ['PubMed-Pegasus', 'Ms-Marco-Pegasus', 'PubMed']

shortened_model_names = ['MS-Marco', 'Ms-Marco-Titles', 'Ms-Marco-Pegasus', 'MsMarco masked model']

shortened_model_names = ['Ms-Marco-Pegasus', 'PubMed', 'Pubmed+Pegasus']

for metric in metrics:
    plt.figure(figsize=(12, 6))
    values = [model_metrics[metric] for model_metrics in results.values()]
    bars = plt.bar(shortened_model_names, values, color='steelblue', alpha=0.7)

    # Adding value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom', ha='center')

    # Setting labels and title
    plt.xlabel('Models', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f'Comparison of {metric} Across Different Models', fontsize=16)

    # Improving x-axis labels readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)

    # Adding a layout for better spacing and display
    plt.tight_layout()
    plt.show()
    
    



metrics = list(results[next(iter(results))].keys())

# Setting a Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Creating the plots with improved colors
for metric in metrics:
    plt.figure(figsize=(12, 6))
    values = [model_metrics[metric] for model_metrics in results.values()]

    # Using a more visually appealing color palette
    bars = plt.bar(shortened_model_names, values, color=sns.color_palette("husl", len(shortened_model_names)))

    # Adding value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom', ha='center')

    # Setting labels and title
    plt.xlabel('Models', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f'Comparison of {metric} Across Different Models', fontsize=16)

    # Improving x-axis labels readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)

    # Adding a layout for better spacing and display
    plt.tight_layout()
    plt.show()














fields = ['mean_embedding', 'vector_abstract1','vector_abstract2']    
fields = 'mean_embedding'

# res_field = {}
# for field in fields:    
results = {}
for model in models:
    index_name, model_name = next(iter(model.items()))
    a = Custom_model_Evaluation(model_name,query_to_doi_dict, index_name.lower(), filter_by_title = False,field = fields, k=5)
    b = Custom_model_Evaluation(model_name,query_to_doi_dict, index_name.lower(), filter_by_title = True,field = fields, k=5)
# Custom_model_Evaluation(model_name,query_to_doi_dict,index_name, filter_by_title = False, field = 'mean_embedding', k=5)
a
b
