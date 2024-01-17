from sentence_transformers import SentenceTransformer, util
import time
import pickle
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import torch
from sentence_transformers import InputExample


if torch.cuda.is_available():
    print("GPU is available. Current device:", torch.cuda.current_device())
else:
    print("GPU is not available. Using CPU.")

    """_summary_
Attempt to train a naive msmarco-distilbert-cos-v5 with the generated questions with pegasus,
to later evaluate the cosine similarity performance on BIER datasets. 
    """
    
def get_summary_from_file(file):    
    if not isfile(join(pegasus_folder, file)):
        print('Not a file')
        return None, None    
    with open(pegasus_folder+file, 'rb') as f:
        try:
            data = pickle.load(f)
            f.close() 
        except:
            print('Error in file')
            return None, None  
    doi = list(data.keys())[0]
    query = list(data.values())[0]
    oa_file = join(oa_folder, doi.replace('/', '_')+'.txt')
    if not isfile(oa_file):
        print('Not a OA file')
    with open(oa_file, 'rb') as f:
        data = pickle.load(f)
        f.close() 
    abstract = data['pubmed']['abstract']        
    return query, abstract


oa_folder = '/home/panchojasen/Data/OA_JSON/'
pegasus_folder = "/home/panchojasen/Data/query_pegasus/"
queries = []
abstracts = []
counter = 0
for file in tqdm(listdir(pegasus_folder)):
    # counter += 1
    query, abstract = get_summary_from_file(file)
    if not abstract:
        continue
    if len(abstract) > 100 and len(query) > 15 and len(abstract) < 3000 and len(query) < 500:
        queries.append(query)
        abstracts.append(abstract)
        counter += 1
        # print(len(query), len(abstract))
        # if counter == 332:
        #     break
    # if counter > 100:
    #     break
print(len(queries)    )
    




model_name = 'pritamdeka/S-PubMedBert-MS-MARCO'
model = SentenceTransformer(model_name)
from torch.utils.data import DataLoader
from sentence_transformers import losses
from torch.optim import AdamW
from torch.optim import AdamW
from transformers import AutoTokenizer


# Create a list of InputExamples - for unsupervised learning, you might use the same abstract as both sentences in the pair
training_examples = [InputExample(texts=[title, text]) for title, text in zip(queries, abstracts)]

# DataLoader for training
train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=5)

# Use a contrastive loss (like MultipleNegativesRankingLoss) for unsupervised learning
train_loss = losses.MultipleNegativesRankingLoss(model)
optimizer_class = AdamW
optimizer_params = {
    'lr': 2e-5,
    'eps': 1e-8
}
nepoch = 5
# Training loop
model.fit(train_objectives=[(train_dataloader, train_loss)], optimizer_class=optimizer_class, optimizer_params=optimizer_params, show_progress_bar=True, epochs=nepoch, warmup_steps=100)
model_output_dir = "./Model_Pubmed_pegasus_allqueries"
# model.save("test_model_pegasus_queries_250k")
model.save(os.path.join(model_output_dir))

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_output_dir)

print(f"Model and tokenizer saved in {model_output_dir}")
