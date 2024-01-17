
##3 NOT WORKING
from elasticsearch_dsl import Document, DenseVector, Text, Keyword, Index, connections, Nested
from elasticsearch_dsl.query import MatchAll
from os import listdir
from os.path import isfile, join
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
from elasticsearch import Elasticsearch
connections.create_connection(hosts=['localhost'], timeout=20)
class MyDocument(Document):
    doi = Text()  # Adding a field for the DOI
    abstract = Text()  # Adding a field for the abstract
    vector_abstract1 = DenseVector(dims=768)
    vector_abstract2 = DenseVector(dims=768)  # Optional fields
    vector_abstract3 = DenseVector(dims=768)
    vector_title = DenseVector(dims=768)
    mean_embedding = DenseVector(dims=768)
    title = Text()

    class Index:
        settings = {'number_of_shards': 1, 'number_of_replicas': 0}

    @classmethod
    def create(cls, database_name, **kwargs):
        cls._index._name = database_name
        return cls(**kwargs)
class MyDocument(Document):
    doi = Text()  # Adding a field for the DOI
    abstract = Text()  # Adding a field for the abstract
    vector_abs = DenseVector(dims=768)
    vector_title = DenseVector(dims=768)
    title = Text()
    class Index:
        name = 'delate2'
        settings = {'number_of_shards': 1, 'number_of_replicas': 0}
  


es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# Step 3: Create the index
database_name = 'msmarco-distilbert-cos-v5'
doc = MyDocument.create(database_name)

# es.indices.delete(index=database_name, ignore=[400, 404])

model = 'sentence-transformers/msmarco-distilbert-cos-v5'

folder = '/home/panchojasen/Data/Vectors_NN/msmarco-distilbert-cos-v5/'
for file in tqdm(Path(folder).iterdir()):
    doc1= None
    with open(file, 'rb') as f:
            doc = pickle.load(f)
            f.close()
    doc['mean_embedding'] = list(np.mean(doc['vector_abstract'], axis=0))
    doc['vector_title'] = list(doc['vector_title'][0])
    doc['vector_abstract1'] = list(doc['vector_abstract'][0])
    if len(doc['vector_abstract'])>1:
        doc['vector_abstract2'] = doc['vector_abstract'][1]
    if len(doc['vector_abstract'])>2:
        doc['vector_abstract3'] = doc['vector_abstract'][2]
    # if len(doc['vector_abstract']) == 2:
    #     doc1 = MyDocument( doi=doc['doi'], abstract=doc['abstract'],mean_embedding = doc['mean_embedding'],
    #                   title=doc['title'], vector_title =doc['vector_title'], vector_abstract1 =  doc['vector_abstract1'],
    #                    vector_abstract2 =  doc['vector_abstract2'])
    # if len(doc['vector_abstract']) > 2:
    #     doc1 = MyDocument( doi=doc['doi'], abstract=doc['abstract'],mean_embedding = doc['mean_embedding'],
    #                   title=doc['title'], vector_title =doc['vector_title'], vector_abstract1 =  doc['vector_abstract1'],
    #                    vector_abstract2 =  doc['vector_abstract2'],vector_abstract3 =  doc['vector_abstract3'])
    # else:
    doc1 = MyDocument( doi=doc['doi'], abstract=doc['abstract'],mean_embedding = doc['mean_embedding'],
                    title=doc['title'], vector_title =doc['vector_title'], vector_abstract1 =  doc['vector_abstract1'])
                #   vector_abstract = doc['vector_abstract'])# doc1 = MyDocument(doc)
    doc1.save()
    # doc1.save()

        
    # break
    
# from elasticsearch_dsl import Search

# s = Search(index=database_name).filter()

# # Execute the search
# response = s.execute()

# # Check if the document is found
# if response.hits.total.value > 0:
#     print("Document found.")
# else:
#     print("Document not found.")