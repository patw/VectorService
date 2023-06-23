from fastapi import FastAPI
import spacy
from sklearn.preprocessing import normalize
import numpy
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import euclidean
import torch
from transformers import BertModel, BertTokenizer, AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

# Load the spacy model that you have installed
nlp_small = spacy.load('en_core_web_sm')
nlp_medium = spacy.load('en_core_web_md')
nlp_lg = spacy.load('en_core_web_lg')

# Load a huggingface 384d sentence transformer
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load BERT sentence model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Splade Model
model_type_or_dir = "naver/splade-cocondenser-ensembledistil"

splade_model = Splade(model_type_or_dir, agg="max")
splade_model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

# Load a huggingface 384d sentence transformer
st_768_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Instructor model
instructor_model = INSTRUCTOR('hkunlp/instructor-large')

print("Preloaded big files")