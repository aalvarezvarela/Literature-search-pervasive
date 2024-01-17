import sys
sys.path.append('/home/panchojasen/Projects/HoneyComb/Common_Functions/')
# from Funtions_query_ElasticSearch import *
from os import listdir
from os.path import isfile, join
from pathlib import Path

import pickle

def get_data_JSON(oa_folder,file, cat):
    with open(oa_folder+file, 'rb') as f:
        data = pickle.load(f)
        
    return data[cat]

def embbebd(text,give_average = False):
    text=clean_text_embedings(text)
    if not isinstance(text, list):
        tokens = tokenizer.tokenize(text)
        # print('this is amount of tokens', len(tokens))
        if len(tokens) > 650:
            chunks = chunk_text(text)
            embedding_abs_list =model.encode(chunks,batch_size=batch_s, convert_to_tensor=False, show_progress_bar=False)
            if give_average:
                embedding = [np.mean(embedding_abs_list, axis=0)]
                return [embedding]
            else:
                return embedding_abs_list  
    embeddings = model.encode(text, batch_size=batch_s,convert_to_tensor=False, show_progress_bar=False)
    return [embeddings]

def clean_text_embedings(text):
    if isinstance(text, list):
        listtext = [clean_text_embedings(sentence) for sentence in text ]
        return listtext
    text = text.replace('..', '.').replace('\n', ' ')
    text = ' '.join([word.strip() for word in text.split(' ') if len(word) > 0])
    return text
def chunk_text(text, max_size=470):
    # Split the entire text into sentences
    sentences = text.split(".")
    chunks = []
    current_chunk_sentences = []
    current_chunk_size = 0
    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        sentence_length = len(sentence_tokens)
        
        # If adding the current sentence exceeds max_size
        if current_chunk_size + sentence_length > max_size:
            chunks.append(".".join(current_chunk_sentences) + ".")
            current_chunk_sentences = [sentence]
            current_chunk_size = sentence_length
        else:
            current_chunk_sentences.append(sentence)
            current_chunk_size += sentence_length

    # Add the remaining sentences as a chunk
    if current_chunk_sentences:
        chunks.append(".".join(current_chunk_sentences) + ".")
      
    return chunks

# a= embbebd(text)
def join_with_comma_and(lst):
    if len(lst) == 0:
        return ""
    elif len(lst) == 1:
        return lst[0]
    else:
        # Join all elements except the last one with ', '
        joined_elements = ', '.join(lst[:-1])
        # Add 'and' before the last element
        return f"{joined_elements} and {lst[-1]}"
    



def save_json_pickle(path, paper):
    with open(path, 'wb') as file:
        pickle.dump(paper, file)

def get_vector_paper(oa_folder,file,save_folder, onlyAbstract = False):
    # s = time.time()
    if onlyAbstract:
        data = file
    else:
        with open(oa_folder+file, 'rb') as f:
            data = pickle.load(f)
            f.close()
    doi = data['doi']
    title = data['title']
    if onlyAbstract:
        abstract = data['abstract']
    else:
        abstract = data.get('pubmed', {}).get('abstract', None)
    paper = {}
    paper['doi'] = doi
    paper['title'] = title
    paper['abstract'] = abstract
    paper['vector_abstract']  = embbebd(abstract)  
    paper['vector_title']  = embbebd(title)  
    save_json_pickle(save_folder+doi.replace('/', '_').lower()+'.txt', paper)
    # print(round(time.time()-s,1),' seconds.', round(i*100/total_len,4), '%')
    return ''


import os
from sentence_transformers import SentenceTransformer, util
import time
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
# model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
models = []
models.append({'pritamdeka/S-PubMedBert-MS-MARCO':'/home/panchojasen/Projects/NN-Engene/Code/Model_Pubmed_pegasus_allqueries'})
models.append({'sentence-transformers/msmarco-distilbert-cos-v5':'/home/panchojasen/Projects/NN-Engene/Code/Model_MSMARCO_pegasus'})
models.append({'sentence-transformers/msmarco-distilbert-cos-v5':'/home/panchojasen/Projects/PaperScraper/Sentence-transformers/test_model_all'})
models.append({'sentence-transformers/msmarco-distilbert-cos-v5':'/home/panchojasen/Projects/PaperScraper/Sentence-transformers/test_model2'})

models.append({'sentence-transformers/msmarco-distilbert-cos-v5':'/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/msmarco-distilbert-cos-v5-finetuned-test-delate'})
models.append({'sentence-transformers/msmarco-distilbert-cos-v5':'/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/Model_pubmed_abstracts'})

models.append({'pritamdeka/S-PubMedBert-MS-MARCO': 'pritamdeka/S-PubMedBert-MS-MARCO'})
models.append({'sentence-transformers/msmarco-distilbert-cos-v5': 'sentence-transformers/msmarco-distilbert-cos-v5'})


pickle_file_path= '/home/panchojasen/Projects/NN-Engene/dois_to_vector/abstracts_evaluation_dois.pkl'

with open(pickle_file_path, 'rb') as file:
    total_dois = pickle.load(file)


for model in models:
    
    model_name = list(model.values())[0]
    full_model_name = list(model.keys())[0]

    # model_name = 'msmarco-distilbert-cos-v5'
    # full_model_name= 'sentence-transformers/msmarco-distilbert-cos-v5'
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    oa_folder = '/home/panchojasen/Data/OA_JSON/'
    save_folder= '/home/panchojasen/Data/Vectors_NN/'
    last_folder_name = os.path.basename(os.path.normpath(model_name))
    last_folder_name
    # Construct the full path for the model folder
    model_folder = os.path.join(save_folder, last_folder_name+'/')

    # Check if the folder exists
    if not os.path.exists(model_folder):
    # Create the folder if it does not exist
        os.makedirs(model_folder)
        print(f"Folder '{model_folder}' created.")
    else:
        print(f"Folder '{model_folder}' already exists.")
# dois_done = [f.replace('.txt', '').replace('_', '/') for f in listdir(model_folder) if isfile(join(model_folder, f))]

    dois_done_names = [f for f in listdir(model_folder) if isfile(join(model_folder, f))]
    with open('/home/panchojasen/Projects/NN-Engene/dois_to_vector/random_dois.pkl', 'rb') as file:
        files_to_do = pickle.load(file)

# for i,file in enumerate(files_to_do):
    
    files_to_do = list(set(files_to_do) - set(dois_done_names))

    batch_s = 128

    for file in tqdm(files_to_do):
        get_vector_paper(oa_folder,file,model_folder)
    for file in tqdm(total_dois):

        get_vector_paper(oa_folder,total_dois[file],model_folder, True)


len(total_dois)








# for i,file in enumerate(files_to_do):
#     tokenizer = AutoTokenizer.from_pretrained('pritamdeka/S-PubMedBert-MS-MARCO')
#     text_abst = 'The abstract of the research article is the following: ' + doc['abstract']
#     tokens = tokenizer.tokenize(text_abst)
#     print('this is amount of tokens', len(tokens))

    
#     embedding_abs = model.encode(text_abst.replace('..', '.'), convert_to_tensor=True)
#     text_title = 'The title of the research article is' + doc['title']
#     embedding_title = model.encode(text_title.replace('..', '.'), convert_to_tensor=True)
#     keywords= []
#     for n in data['concepts']:
#         keywords += [n['display_name']]

#     for n in data['mesh']:
#         keywords += [n['descriptor_name'].replace(', Major', '').replace(', Minor', '')]

#     text_keywords = 'The main keywords and concepts of the research article are '
#     text_keywords += ', '.join(keywords[:-1]) + ', and ' + keywords[-1]
#     embedding_keywords = model.encode(text_keywords.replace('..', '.'), convert_to_tensor=True)

#     res = MyDocument(doi=doc['doi'], title=doc['title'], abstract=doc['abstract'], vector_abs=embedding_abs.tolist(), vector_title = embedding_title.tolist(), vector_keywords = embedding_keywords.tolist())
#     res.save()
#     e = time.time()
#     print('Time (sec) lasted:',e-s)
#     # return
#     # except:
#     #     print('error, skipping')
#     #     return






















# # Step 2: Define a Document subclass with a dense vector field

# from elasticsearch_dsl import Document, DenseVector, Text, Keyword, Index, connections
# from elasticsearch_dsl.query import MatchAll
# connections.create_connection(hosts=['localhost'], timeout=20)

# class MyDocument(Document):
#     doi = Text()  # Adding a field for the DOI
#     abstract = Text()  # Adding a field for the abstract
#     vector_abs = DenseVector(dims=768)
#     vector_title = DenseVector(dims=768)
#     vector_keywords = DenseVector(dims=768)
#     title = Text()
#     class Index:
#         name = 'delate'
#         settings = {'number_of_shards': 1, 'number_of_replicas': 0}

# # Step 3: Create the index
# MyDocument.init()

# # Step 4: Index some documents
# # doc1 = MyDocument(meta={'id': 1}, doi=doc['doi'], abstract=doc['abstract'], vector=embedding_1.tolist())
# # es.indices.delete(index='delate', ignore=[400, 404])
# # doc1.save()

# # es.indices.delete(index='delate')

# from tqdm import tqdm

# for file in tqdm(Path(oa_folder).iterdir()):
#     # if file.name.replace('_','/').replace('.txt', '') == '10.1038/s43018-022-00402-0':
#     #     print('noshit')
#     #     break
#     if file.name.replace('_','/').replace('.txt', '') not in dois_e+dois_hans:
#         print(file.name)
#         add_papers_es(file)
        
#     # break

# ### Perform the query
# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# query = 'Notch singalling in breast cancer'
# query_e = model.encode(query, convert_to_tensor=True)
# query_vector = query_e.tolist()
# # Perform a script score query to find the nearest neighbors
# response = es.search(index="delate", body={
#     "size": 100,  # Add this line to request 20 documents
#     "query": {
#         "script_score": {
#             "query": {"match_all": {}},
#             "script": {
#                 "source": "cosineSimilarity(params.query_vector, 'vector_abs') + 1.0",
#                 "params": {"query_vector": query_vector}
#             }
#         }
#     }
# })

# ids = []
# for hit in response['hits']['hits']:
#     ids += [hit['_id']]
#     # print(f"Document (score: {hit['_score']}):{hit['_source']['doi']} {hit['_source']['title']}")



# response = es.search(index="delate", body={
#     "size": 25,  # Request 20 documents
#     "query": {
#         "bool": {
#             "must": {
#                 "script_score": {
#                     "query": {"match_all": {}},
#                     "script": {
#                         "source": "cosineSimilarity(params.query_vector, 'vector_title') + 1.0",
#                         "params": {"query_vector": query_vector}
#                     }
#                 }
#             },
#             "filter": [
#                 {"ids": {"values": ids}}
#             ]
#         }
#     }
# })
# for hit in response['hits']['hits']:
#     print(f"Document (score: {hit['_score']}):{hit['_source']['doi']} {hit['_source']['title']}")

