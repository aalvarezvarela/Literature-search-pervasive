"""
Create a class to compare models
"""

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoModel
import logging
import pathlib, os
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer


class YourCustomDEModel:
    def __init__(self, model_path=None, **kwargs):
        # Load your custom model
        self.model = SentenceTransformer(model_path) if model_path is not None else None

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        # Encode the queries using the SentenceTransformer's encode method
        # and return the embeddings as a numpy array
        if self.model is None:
            raise ValueError("Model not loaded. Please specify a model path when initializing.")
        embeddings = self.model.encode(queries, batch_size=batch_size, **kwargs)
        return embeddings.cpu().numpy()  # Move to CPU and convert to numpy array

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        # Extract texts from the corpus and encode them
        # Corpus is expected to be a list of dictionaries where each dictionary has a 'text' field
        if self.model is None:
            raise ValueError("Model not loaded. Please specify a model path when initializing.")
        corpus_texts = [doc['text'] for doc in corpus]
        embeddings = self.model.encode(corpus_texts, batch_size=batch_size, **kwargs)
        return embeddings.cpu().numpy()


# def model_comparison(model,model_aka, model_dir = None,dataset = 'scidocs', bs=128):
    
def model_comparison(model_dir,model_aka, custom = False, dataset = 'scidocs', bs=10):
    model_results = {}
    if dataset not in ['scidocs', 'scifact']:
        raise 'Only scidocs / scifact accepted'
    
#### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    
    #Download_dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
#### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    if custom:
        print('!!!!! LOADING CUSTOM MODE')
        custom_model = YourCustomDEModel(model_dir)
        model_to_ev = DRES(custom_model, batch_size=bs, corpus_chunk_size=512*9999)
    #### Load the SBERT model and retrieve using cosine-similarity
    else:
        model_to_ev = DRES(models.SentenceBERT(model_dir), batch_size=bs, corpus_chunk_size=512*9999)
    retriever = EvaluateRetrieval(model_to_ev, score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)
    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    model_results[model_aka] = {
            "NDCG": ndcg,
            "MAP": _map,
            "Recall": recall,
            "Precision": precision
        }
    return model_results

ev_results = {}

model_name = "sentence-transformers/msmarco-distilbert-cos-v5"

resu = model_comparison(model_name,'msmarco-distilbert-cos-v5', custom = False)

ev_results.update(resu)




# model_path= "/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/msmarco-distilbert-cos-v5-finetuned-test-delate"
model_name2= "/home/panchojasen/Projects/NN-Engene/Code/Model_MSMARCO_pegasus/"
# model_path = "/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/Model_pubmed_abstracts/"
resu2 = model_comparison(model_name2,'pegasus', custom = True)
ev_results.update(resu2)

model_path5= "pritamdeka/S-PubMedBert-MS-MARCO"
resu5 =  model_comparison(model_path5,'PubMed', custom = False)
ev_results.update(resu5)

model_path6= "/home/panchojasen/Projects/NN-Engene/Code/Model_Pubmed_pegasus_allqueries"
resu6 =  model_comparison(model_path6,'PubMed_allqueries', custom = True)
ev_results.update(resu6)

model_path3= "/home/panchojasen/Projects/PaperScraper/Sentence-transformers/test_model_all"

resu3 =  model_comparison(model_path3,'test_model_all', custom = True)
ev_results.update(resu3)

model_path4= "/home/panchojasen/Projects/PaperScraper/Sentence-transformers/test_model2"
resu4 =  model_comparison(model_path4,'test_model_all2', custom = True)
ev_results.update(resu4)


model_path5= "/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/Model_pubmed_abstracts"
resu5 =  model_comparison(model_path5,'Model_pubmed_abstracts', custom = True)
ev_results.update(resu5)

model_path6= "/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/msmarco-distilbert-cos-v5-finetuned-test-delate"
resu6 =  model_comparison(model_path6,'test_delate', custom = True)
ev_results.update(resu6)





import matplotlib.pyplot as plt

def plot_selected_metric(results, metric_name, n=1):
    if metric_name == 'Precision':
        aka_name = 'P'
    else:
        aka_name = metric_name
    # Extracting the selected metric
    selected_metric = {}
    for model, metrics in results.items():
        if metric_name in metrics:
            selected_metric[model] = {f"{aka_name}@{n}": metrics[metric_name].get(f"{aka_name}@{n}", 0)}

    # Plotting
    plt.figure(figsize=(6, 4))
    models = list(selected_metric.keys())
    values = [list(metrics.values())[0] for metrics in selected_metric.values()]
    
    plt.bar(models, values)
    plt.title(f"{metric_name}@{n} Comparison")
    plt.ylim(0, 1)
    for index, value in enumerate(values):
        plt.text(index, value + 0.02, f"{value:.2f}", ha='center')

    plt.tight_layout()
    plt.show()

    # Plotting each metric individually
plot_selected_metric(ev_results, 'NDCG', 10)
plot_selected_metric(ev_results, 'MAP', 10)
plot_selected_metric(ev_results, 'Recall',10)
plot_selected_metric(ev_results, 'Precision',10)
import json
with open('/home/panchojasen/Projects/NN-Engene/Code/Validation_scidocs.json', 'w') as json_file:
    json.dump(ev_results, json_file)