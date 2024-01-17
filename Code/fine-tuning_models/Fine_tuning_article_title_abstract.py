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


model_name = 'msmarco-distilbert-cos-v5'

model = SentenceTransformer(f'sentence-transformers/{model_name}')
oa_folder = '/home/panchojasen/Data/OA_JSON/'
print(next(model.parameters()).device)

def get_abstract_title_paper(oa_folder,file,save_folder):
    # s = time.time()
    with open(oa_folder+file, 'rb') as f:
        data = pickle.load(f)
        f.close() 
    try:
        abstract = data['pubmed']['abstract']
        title = data['pubmed']['title']
        return title, abstract
    except:
         return None, None

from torch.utils.data import DataLoader
from sentence_transformers import losses
from torch.optim import AdamW
from torch.optim import AdamW



# files = [f for f in listdir(oa_folder) if isfile(join(oa_folder, f))]

titles = []
abstracts = []
counter = 0
for file in tqdm(listdir(oa_folder)):
    # try:
        if not isfile(join(oa_folder, file)):
            print('Not a file')
            continue    
        if counter == 3:
             counter = 0
        if counter > 0:
             counter += 1
             continue
        if len(abstracts) > 250_000:
             break
        title, abstract = get_abstract_title_paper(oa_folder,file,'save_folder')
        if title != None and abstract != None:
            titles.append(title)
            abstracts.append(abstract)
            counter += 1


from transformers import AutoTokenizer
old_tokenizer =  AutoTokenizer.from_pretrained(f'sentence-transformers/{model_name}')
tokenizer = old_tokenizer.train_new_from_iterator(abstracts, 50000)

# model[0].tokenizer = tokenizer




# Create a list of InputExamples - for unsupervised learning, you might use the same abstract as both sentences in the pair
training_examples = [InputExample(texts=[title, text]) for title, text in zip(titles, abstracts)]

# DataLoader for training
train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=16)

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

# model.save("test_model_titles_250k")

# model2 = SentenceTransformer('/home/panchojasen/Projects/PaperScraper/Sentence-transformers/test_model_all')



# abstract1 = 'Colorectal cancer (CRC) patient-derived organoids predict responses to chemotherapy. Here we used them to investigate relapse after treatment. Patient-derived organoids expand from highly proliferative LGR5+ tumor cells; however, we discovered that lack of optimal growth conditions specifies a latent LGR5+ cell state. This cell population expressed the gene MEX3A, is chemoresistant and regenerated the organoid culture after treatment. In CRC mouse models, Mex3a+ cells contributed marginally to metastatic outgrowth; however, after chemotherapy, Mex3a+ cells produced large cell clones that regenerated the disease. Lineage-tracing analysis showed that persister Mex3a+ cells downregulate the WNT/stem cell gene program immediately after chemotherapy and adopt a transient state reminiscent to that of YAP+ fetal intestinal progenitors. In contrast, Mex3a-deficient cells differentiated toward a goblet cell-like phenotype and were unable to resist chemotherapy. Our findings reveal that adaptation of cancer stem cells to suboptimal niche environments protects them from chemotherapy and identify a candidate cell of origin of relapse after treatment in CRC. '
# # abstract2 = 'Colorectal Cancer Cells Enter a Diapause-like DTP State to Survive Chemotherapy'
# # abstract3 = 'Large-scale pancreatic cancer detection via non-contrast CT and deep learning'
# # abstract4 = 'Zonation of Ribosomal DNA Transcription Defines a Stem Cell Hierarchy in Colorectal Cancer'
# abstract5 = 'Colorectal cancer cells expressing Mex3a drive recurrence after chemotherapy'
# abstract6 = 'Mex3a marks drug-tolerant persister colorectal cancer cells that mediate relapse after chemotherapy'
#
# 
#  abstract1 = 'Colorectal cancer (CRC) patient-derived organoids predict responses to chemotherapy. Here we used them to investigate relapse after treatment. Patient-derived organoids expand from highly proliferative LGR5+ tumor cells; however, we discovered that lack of optimal growth conditions specifies a latent LGR5+ cell state. This cell population expressed the gene MEX3A, is chemoresistant and regenerated the organoid culture after treatment. In CRC mouse models, Mex3a+ cells contributed marginally to metastatic outgrowth; however, after chemotherapy, Mex3a+ cells produced large cell clones that regenerated the disease. Lineage-tracing analysis showed that persister Mex3a+ cells downregulate the WNT/stem cell gene program immediately after chemotherapy and adopt a transient state reminiscent to that of YAP+ fetal intestinal progenitors. In contrast, Mex3a-deficient cells differentiated toward a goblet cell-like phenotype and were unable to resist chemotherapy. Our findings reveal that adaptation of cancer stem cells to suboptimal niche environments protects them from chemotherapy and identify a candidate cell of origin of relapse after treatment in CRC. '

# abstract2 = 'Cancer cells enter a reversible drug-tolerant persister (DTP) state to evade death from chemotherapy and targeted agents. It is increasingly appreciated that DTPs are important drivers of therapy failure and tumor relapse. We combined cellular barcoding and mathematical modeling in patient-derived colorectal cancer models to identify and characterize DTPs in response to chemotherapy. Barcode analysis revealed no loss of clonal complexity of tumors that entered the DTP state and recurred following treatment cessation. Our data fit a mathematical model where all cancer cells, and not a small subpopulation, possess an equipotent capacity to become DTPs. Mechanistically, we determined that DTPs display remarkable transcriptional and functional similarities to diapause, a reversible state of suspended embryonic development triggered by unfavorable environmental conditions. Our study provides insight into how cancer cells use a developmentally conserved mechanism to drive the DTP state, pointing to novel therapeutic opportunities to target DTPs.'

# abstract3 = 'Pancreatic ductal adenocarcinoma (PDAC), the most deadly solid malignancy, is typically detected late and at an inoperable stage. Early or incidental detection is associated with prolonged survival, but screening asymptomatic individuals for PDAC using a single test remains unfeasible due to the low prevalence and potential harms of false positives. Non-contrast computed tomography (CT), routinely performed for clinical indications, offers the potential for large-scale screening, however, identification of PDAC using non-contrast CT has long been considered impossible. Here, we develop a deep learning approach, pancreatic cancer detection with artificial intelligence (PANDA), that can detect and classify pancreatic lesions with high accuracy via non-contrast CT. PANDA is trained on a dataset of 3,208 patients from a single center. PANDA achieves an area under the receiver operating characteristic curve (AUC) of 0.986-0.996 for lesion detection in a multicenter validation involving 6,239 patients across 10 centers, outperforms the mean radiologist performance by 34.1% in sensitivity and 6.3% in specificity for PDAC identification, and achieves a sensitivity of 92.9% and specificity of 99.9% for lesion detection in a real-world multi-scenario validation consisting of 20,530 consecutive patients. Notably, PANDA utilized with non-contrast CT shows non-inferiority to radiology reports (using contrast-enhanced CT) in the differentiation of common pancreatic lesion subtypes. PANDA could potentially serve as a new tool for large-scale pancreatic cancer screening. '

# abstract4 = 'Colorectal cancers (CRCs) are composed of an amalgam of cells with distinct genotypes and phenotypes. Here, we reveal a previously unappreciated heterogeneity in the biosynthetic capacities of CRC cells. We discover that the majority of ribosomal DNA transcription and protein synthesis in CRCs occurs in a limited subset of tumor cells that localize in defined niches. The rest of the tumor cells undergo an irreversible loss of their biosynthetic capacities as a consequence of differentiation. Cancer cells within the biosynthetic domains are characterized by elevated levels of the RNA polymerase I subunit A (POLR1A). Genetic ablation of POLR1A-high cell population imposes an irreversible growth arrest on CRCs. We show that elevated biosynthesis defines stemness in both LGR5+ and LGR5- tumor cells. Therefore, a common architecture in CRCs is a simple cell hierarchy based on the differential capacity to transcribe ribosomal DNA and synthesize proteins.'
# word1 = 'non proliferative cell'
# word2 ='label retaining cell'
# word3 = 'colon'
# word4 = 'stem cell'
# word5 = 'quiescent cell'
# query = 'Paper studying chemotherapy in colon cancer'

# sentences = [abstract1, abstract2, abstract3, abstract4,abstract5,abstract6]
# sentences = [word1, word2, word3, word4, word5]
# sentences = [query, abstract1, abstract2, abstract3, abstract4]

# from sklearn.metrics.pairwise import cosine_similarity
# embeddings1 = model.encode(sentences)
# cosine_similarities1 = cosine_similarity([embeddings1[0]], embeddings1)
# print(cosine_similarities1)
# # Print the cosine similarities
# embeddings2 = model2.encode(sentences)
# cosine_similarities2 = cosine_similarity([embeddings2[0]], embeddings2)
# print(cosine_similarities2)


# from transformers import AutoTokenizer
# old_tokenizer =  AutoTokenizer.from_pretrained(f'sentence-transformers/{model_name}')
# tokenizer = old_tokenizer.train_new_from_iterator(abstracts, 52000)

# model[0].tokenizer = tokenizer