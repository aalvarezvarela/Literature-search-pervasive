from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import List, Dict
import numpy as np
import logging
import pathlib, os
import random

# Custom import for loading models
from sentence_transformers import SentenceTransformer

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
class CustomSentenceTransformer(SentenceTransformer):
    def encode_queries(self, queries, batch_size=32, show_progress_bar=True, **kwargs):
        # Encode the queries using the SentenceTransformer's encode method
        return self.encode(queries, batch_size=batch_size, show_progress_bar=show_progress_bar, **kwargs)

    def encode_corpus(self, corpus, batch_size=32, show_progress_bar=True, **kwargs):
        # Encode the corpus using the SentenceTransformer's encode method
        return self.encode(corpus, batch_size=batch_size, show_progress_bar=show_progress_bar, **kwargs)


class CustomSentenceTransformer(SentenceTransformer):
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        pass
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        pass
    
    
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
# model_path = 'path/to/your/model'
# custom_model = YourCustomDEModel(model_path=model_path)    

dataset = "scidocs"

#### Download dataset and unzip the dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Dense Retrieval using custom model ####

# Path to the custom model or model name if it's a pre-trained model available in sentence-transformers
custom_model_path = "/home/panchojasen/Projects/NN-Engene/Code/Model_MSMARCO_pegasus"
# custom_model_path= "/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/msmarco-distilbert-cos-v5-finetuned-test-delate"
custom_model_path= "/home/panchojasen/Projects/PaperScraper/Sentence-transformers/test_model_all"

# Load the custom model
custom_model = YourCustomDEModel(custom_model_path)
model = DRES(custom_model, batch_size=10, corpus_chunk_size=512*9999)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

#### Retrieve dense results (format of results is identical to qrels)
start_time = time()
results = retriever.retrieve(corpus, queries)
end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)


model = DRES(models.SentenceBERT("sentence-transformers/msmarco-distilbert-cos-v5"), batch_size=64, corpus_chunk_size=512*9999)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

#### Retrieve dense results (format of results is identical to qrels)
start_time = time()
results = retriever.retrieve(corpus, queries)
end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg2, _map2, recall2, precision2 = retriever.evaluate(qrels, results, retriever.k_values)


# Function to print metrics side by side
def print_metrics_side_by_side(metrics1, metrics2):
    for metric in metrics1:
        print(f"Metric: {metric}")
        print(f"{'K':<10} {'Model 1':<15} {'Model 2':<15}")
        print(f"{'-'*40}")
        for k, v in metrics1[metric].items():
            value2 = metrics2[metric].get(k, 'N/A')  # Get corresponding value from model 2
            print(f"{k:<10} {v:<15.5f} {value2:<15.5f}")
        print()

# Call the function with your metrics
print_metrics_side_by_side([ndcg2, _map2, recall2, precision2 ], [ndcg, _map, recall, precision ])


custom_model_path = "/home/panchojasen/Projects/NN-Engene/Code/Model_MSMARCO_pegasus"
# custom_model_path= "/home/panchojasen/Projects/NN-Engene/Code/fine-tuning_models/msmarco-distilbert-cos-v5-finetuned-test-delate"
# custom_model_path= "/home/panchojasen/Projects/PaperScraper/Sentence-transformers/test_model_all"

# Load the custom model
custom_model = YourCustomDEModel(custom_model_path)
model = DRES(custom_model, batch_size=10, corpus_chunk_size=512*9999)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

#### Retrieve dense results (format of results is identical to qrels)
start_time = time()
results = retriever.retrieve(corpus, queries)
end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg3, _map3, recall3, precision3 = retriever.evaluate(qrels, results, retriever.k_values)
