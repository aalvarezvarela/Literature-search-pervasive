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
import pandas as pd


def Custom_model_Evaluation(model_name,query_to_doi_dict,index_name, filter_by_title = False, field = 'mean_embedding', k=5):
    model = SentenceTransformer(model_name)
    queries = []
    actual = []
    predicted = []
    for query in query_to_doi_dict:
        queries += [query]
        actual += [query_to_doi_dict[query]]
        if filter_by_title:
            predicted += [execute_query_title(query, model,index_name, filter = field)]
        else:
            predicted += [execute_query(query, model,index_name, field)]
    
    mapk_value = mapk(actual, predicted, k=k)
    mrr_value = mean_reciprocal_rank(actual, predicted)
    mean_ndcg_at_k =  sum([ndcg_at_k(actual[i], predicted[i], k=k) for i in range(0,len(actual))]) / len(actual)
    mean_recall_at_k =  sum([recall_at_k(actual[i], predicted[i], k=k) for i in range(0,len(actual))]) / len(actual)
    mean_precision_at_k =  sum([precision_at_k(actual[i], predicted[i], k=k) for i in range(0,len(actual))]) / len(actual)
    mean_frequency_20 = mean_frequency_hits(actual, predicted)
    mean_frequency_10 = mean_frequency_hits(actual, predicted, 10)
    mean_frequency_5 = mean_frequency_hits(actual, predicted, 5)
    print('--------------------------------------------------')
    print("MAP@K:", mapk(actual, predicted, k=k))
    print("MRR:", mean_reciprocal_rank(actual, predicted))
    print("NDCG@K: Mean", mean_ndcg_at_k)
    print("Recall@K: Mean", mean_recall_at_k)
    print("Precision@K: Mean", mean_precision_at_k)
    print('Mean Frequency_20: ',mean_frequency_20)
    print('Mean Frequency_10: ',mean_frequency_10)
    print('--------------------------------------------------')
    print()
 
# Create the dictionary
    metrics_dict ={index_name:{
        "MAP@K": mapk_value,
        "MRR": mrr_value,
        "NDCG@K": mean_ndcg_at_k,
        "Recall@K": mean_recall_at_k,
        "Precision@K": mean_precision_at_k,
        "MeanFreq_20": mean_frequency_20,
        "MeanFreq_10": mean_frequency_10,
        "MeanFreq_5": mean_frequency_5,
    }}
    return metrics_dict




def execute_query(query,model,index_name, field):
    # Replace with your actual Elasticsearch query format
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    query_e = model.encode(query, convert_to_tensor=True)
    query_vector = query_e.tolist()
    if isinstance(field,list):
        script_score_queries = []
        for element in field:
            script_score_queries.append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{element}') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            })
        combined_query = {
            "size": 20,  # Number of documents to retrieve
            "query": {
                "bool": {
                    "should": script_score_queries
                }
            }
        }
        response = es.search(index=index_name, body=combined_query)
# Perform a script score query to find the nearest neighbors
    else:
        response = es.search(index=index_name, body={
        "size": 20,  # Add this line to request 20 documents
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, '{field}') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    })
    if 'persister' in query:
        print(f"query:{query}")
        for hit in response['hits']['hits']:
            print(f"Document (score: {hit['_score']}): {hit['_source']['title']}")
    return [hit['_source']['doi'] for hit in response['hits']['hits']]


# Executing the search query

    
def precision_at_k(actual, predicted, k =10):
    """Calculate Precision at K."""
    predicted = predicted[:k]
    relevant_and_retrieved = len(set(actual) & set(predicted))
    return relevant_and_retrieved / len(predicted)



def recall_at_k(actual, predicted, k = 10):
    """Calculate Recall at K."""
    predicted = predicted[:k]
    relevant_and_retrieved = len(set(actual) & set(predicted))
    return relevant_and_retrieved / len(actual)

def apk(actual, predicted, k=10):
    """Calculate Average Precision at K."""
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """Calculate Mean Average Precision at K."""
    return sum(apk(a, p, k) for a, p in zip(actual, predicted)) / len(actual)

def mean_reciprocal_rank(actual, predicted):
    """Calculate Mean Reciprocal Rank."""
    rr_list = []
    for a, p in zip(actual, predicted):
        for rank, item in enumerate(p, 1):
            if item in a:
                rr_list.append(1 / rank)
                break
        else:
            rr_list.append(0)
    return sum(rr_list) / len(rr_list)

def mean_frequency_hits(actual, predicted, cutoff = None):
    """Calculate Mean Reciprocal Rank."""
    freqs = []
    for a, p in zip(actual, predicted):
        if cutoff:
            p = p[:cutoff]
        # Convert lists to sets for efficient intersection operation
        set_a = set(a)
        set_p = set(p)

        # Calculate the intersection of 'set_a' and 'set_p'
        hits = len(set_a.intersection(set_p))
        freq = hits/len(a)
        freqs += [freq]
    # Calculate the mean frequency
    mean_frequency =sum(freqs)/len(actual)
    return mean_frequency


def ndcg_at_k(actual, predicted, k=10):
    """Calculate Normalized Discounted Cumulative Gain at K."""
    import numpy as np
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

    dcg_max = dcg_at_k(sorted([int(item in actual) for item in predicted], reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k([int(item in actual) for item in predicted], k) / dcg_max




def execute_query_title(query,model,index_name, filter = 'mean_embedding', field2 = 'vector_title'):
    # Replace with your actual Elasticsearch query format
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    query_e = model.encode(query, convert_to_tensor=True)
    query_vector = query_e.tolist()
    if isinstance(filter,list):
        script_score_queries = []
        for element in filter:
            script_score_queries.append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{element}') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            })
        combined_query = {
            "size": 20,  # Number of documents to retrieve
            "query": {
                "bool": {
                    "should": script_score_queries
                }
            }
        }
        filtered_response = es.search(index=index_name, body=combined_query)
# Perform a script score query to find the nearest neighbors
    else:
        filtered_response = es.search(index=index_name, body={
        "size": 40,  # Add this line to request 20 documents
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, '{filter}') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    })
    ids = [hit['_id'] for hit in filtered_response['hits']['hits']]
    # print(ids)
    

    response = es.search(index=index_name, body={
        "size": 20,  # Request 20 documents
        "query": {
            "bool": {
                "must": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, '{field2}') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                },
                "filter": [
                    {"ids": {"values": ids}}
                ]
            }
        }
    })
    if 'persister' in query:
        print(f"query:{query}")
        for hit in response['hits']['hits']:
            print(f"Document (score: {hit['_score']}): {hit['_source']['title']}")

    return [hit['_source']['doi'] for hit in response['hits']['hits']]