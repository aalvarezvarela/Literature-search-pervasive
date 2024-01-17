from elasticsearch_dsl import Document, DenseVector, Text, Keyword, Index, connections, Nested
from elasticsearch_dsl.query import MatchAll
from os import listdir
from os.path import isfile, join
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
from elasticsearch import Elasticsearch

mapping = {
    "mappings": {
        "properties": {
            "doi": {
                "type": "text"
            },
            "abstract": {
                "type": "text"
            },
            "vector_abstract1": {
                "type": "dense_vector",
                "dims": 768
            },
            "vector_abstract2": {
                "type": "dense_vector",
                "dims": 768
            },
            "vector_abstract3": {
                "type": "dense_vector",
                "dims": 768
            },
            "vector_title": {
                "type": "dense_vector",
                "dims": 768
            },
            "mean_embedding": {
                "type": "dense_vector",
                "dims": 768
            },
            "title": {
                "type": "text"
            }
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}
es = Elasticsearch()


import os

# Path to the directory whose folders you want to list
directory_path = "/home/panchojasen/Data/Vectors_NN/"

# List all the subdirectories inside the specified directory
folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]

for folder in folders:
# Define the index name
    index_name = folder.lower()
    index_exists = es.indices.exists(index=index_name)
    if index_exists:
        print(f'folder {folder} exists, delating..')
        es.indices.delete(index=index_name, ignore=[400, 404])
        # continue
    es.indices.create(index=index_name, body=mapping)

    folder = directory_path + folder

    for file in tqdm(Path(folder).iterdir()):
        try:
            with open(file, 'rb') as f:
                doc = pickle.load(f)

            # Convert the entire 'vector_abstract' and 'vector_title' arrays to lists
            if isinstance(doc['vector_abstract'], np.ndarray):
                doc['vector_abstract'] = doc['vector_abstract'].tolist()
            if isinstance(doc['vector_title'], np.ndarray):
                doc['vector_title'] = doc['vector_title'].tolist()

            # Ensure each element in 'vector_abstract' and 'vector_title' is a list
            doc['vector_abstract'] = [vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in doc['vector_abstract']]
            doc['vector_title'] = [vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in doc['vector_title']]

            # Prepare the document for indexing
            document = {
                "doi": doc['doi'].lower(),
                "title": doc['title'],
                "abstract": doc['abstract'],
                "vector_abstract1": doc['vector_abstract'][0] if len(doc['vector_abstract']) > 0 else [],
                "mean_embedding": np.mean(doc['vector_abstract'], axis=0).tolist() if doc['vector_abstract'] else [],
                "vector_title": doc['vector_title'][0] if len(doc['vector_title']) > 0 else []
            }

            # Add additional abstract vectors if available
            document["vector_abstract2"] = doc['vector_abstract'][0]
            document["vector_abstract3"] = doc['vector_abstract'][0]
            if len(doc['vector_abstract']) > 1:
                document["vector_abstract2"] = doc['vector_abstract'][1]
            if len(doc['vector_abstract']) > 2:
                document["vector_abstract3"] = doc['vector_abstract'][2]

            # Index the document
            response = es.index(index=index_name, body=document)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
        






# from sentence_transformers import SentenceTransformer

# model_name = 'sentence-transformers/msmarco-distilbert-cos-v5'
# model = SentenceTransformer(model_name)


# ### Perform the query
# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# query = 'Chemotherapy resistance and heterogeneity in colon cancer'
# query_e = model.encode(query, convert_to_tensor=True)
# query_vector = query_e.tolist()

# # Perform a script score query to find the nearest neighbors
# response = es.search(index=index_name, body={
#     "size": 10,  # Add this line to request 20 documents
#     "query": {
#         "script_score": {
#             "query": {"match_all": {}},
#             "script": {
#                 "source": "cosineSimilarity(params.query_vector, 'vector_abstract') + 1.0",
#                 "params": {"query_vector": query_vector}
#             }
#         }
#     }
# })

# ids = []
# for hit in response['hits']['hits']:

#     ids += [hit['_id']]