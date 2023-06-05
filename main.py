from fastapi import FastAPI
import spacy
from sklearn.preprocessing import normalize
import numpy
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import euclidean
import torch
from transformers import BertModel, BertTokenizer, AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade

# Load the spacy model that you have installed
nlp_small = spacy.load('en_core_web_sm')
nlp_medium = spacy.load('en_core_web_md')
nlp_lg = spacy.load('en_core_web_lg')

# Load a huggingface 384d sentence transformer
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load a huggingface 384d sentence transformer
st_768_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

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

# Fast API init
app = FastAPI(
        title="TextVectorizor",
        description="Create dense vectors using the Spacy English language small (96d), medium (300d) and large (300d) models.  As well as Huggingface Sentence Encoder (384d) and google BERT (768d). Use the similarity tools to compare words and phrases.",
        version="1.0",
        contact={
            "name": "Pat Wendorf",
            "email": "pat.wendorf@mongodb.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }
)

# Perform cosine similarity check between two vectors and output the measurement
def similarity(v1, v2):
    # Define two dense vectors as NumPy arrays
    vector1 = numpy.array(v1)
    vector2 = numpy.array(v2)

    # Compute Euclidean distance
    euclidean_distance = euclidean(vector1, vector2)

    # Compute dot product
    dot_product = numpy.dot(vector1, vector2)

    # Compute cosine similarity
    cosine_similarity = numpy.dot(vector1, vector2) / (numpy.linalg.norm(vector1) * numpy.linalg.norm(vector2))

    return {"euclidean": euclidean_distance, "dotProduct": dot_product, "cosine": cosine_similarity}

# L2 normalization on vectors using sklearn utility function
def vector_normalize(vec):
    shaped = vec.reshape(-1,1)
    normed = normalize(shaped,axis=0)
    return normed.reshape(1,-1)[0]

# Most Similar Word Function (stolen from Spacy github issues!)
def most_similar_words(nlp, word, num_words):
    ms = nlp.vocab.vectors.most_similar(numpy.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]), n=num_words)
    return [nlp.vocab.strings[w] for w in ms[0][0]]

# BERT is more complex than the others...
def bert_nlp(text):
    # Bert wants to batch process, so just slap a single one in here
    inputs = [text]
    input_ids = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors='pt')['input_ids'].to(device)
    attention_mask = (input_ids != 0).to(device)
    outputs = bert_model(input_ids, attention_mask)
    embeddings = outputs.last_hidden_state
    cls_embeddings = embeddings[:, 0, :]
    embedding_list = cls_embeddings.tolist()
    return numpy.array(embedding_list[0])

@app.get("/")
async def root():
    return {"message": "Feed me text and I pop out vector clouds. See /docs for more info."}

@app.get("/svec/")
async def vectorize_text_small(text: str, l2: bool = False):
    doc = nlp_small(text)
    if l2:
        return vector_normalize(doc.vector).tolist()
    else:
        return doc.vector.tolist() 

@app.get("/mvec/")
async def vectorize_text_medium(text: str, l2: bool = False):
    doc = nlp_medium(text)
    if l2:
        return vector_normalize(doc.vector).tolist()
    else:
        return doc.vector.tolist() 

@app.get("/lvec/")
async def vectorize_text_large(text: str, l2: bool = False):
    doc = nlp_lg(text)
    if l2:
        return vector_normalize(doc.vector).tolist()
    else:
        return doc.vector.tolist()
    
@app.get("/stvec/")
async def vectorize_text_st(text: str, l2: bool = False):
    doc = st_model.encode(text)
    if l2:
        return vector_normalize(doc).tolist()
    else:
        return doc.tolist()
    
@app.get("/stvec768/")
async def vectorize_text_sentence_transformer_768(text: str, l2: bool = False):
    doc = st_768_model.encode(text)
    if l2:
        return vector_normalize(doc).tolist()
    else:
        return doc.tolist()
    
@app.get("/bertvec/")
async def vectorize_text_bert(text: str, l2: bool = False):
    doc = bert_nlp(text)
    if l2:
        return vector_normalize(doc).tolist()
    else:
        return doc.tolist()

@app.get("/ssim/")
async def similarity_text_small(t1: str, t2: str):
    d1 = nlp_small(t1)
    d2 = nlp_small(t2)
    return d1.similarity(d2)

@app.get("/msim/")
async def similarity_text_medium(t1: str, t2: str):
    d1 = nlp_medium(t1)
    d2 = nlp_medium(t2)
    return d1.similarity(d2)

@app.get("/lsim/")
async def similarity_text_large(t1: str, t2: str):
    d1 = nlp_lg(t1)
    d2 = nlp_lg(t2)
    return d1.similarity(d2)

@app.get("/msyn/")
async def synonyms_medium(text: str):
    return most_similar_words(nlp_medium, text, 10)

@app.get("/lsyn/")
async def synonyms_large(text: str):
    return most_similar_words(nlp_lg, text, 10)

@app.get("/splade/")
async def splade_bow(text: str):
    with torch.no_grad():
        doc_rep = splade_model(d_kwargs=tokenizer(text, return_tensors="pt"))["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)

    # get the number of non-zero dimensions in the rep:
    col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
    print("number of actual dimensions: ", len(col))

    # now let's inspect the bow representation:
    weights = doc_rep[col].cpu().tolist()
    d = {k: v for k, v in zip(col, weights)}
    sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    bow_rep = []
    for k, v in sorted_d.items():
        bow_rep.append((reverse_voc[k], round(v, 2)))
        
    return bow_rep

@app.get("/simdiff/")
async def similarity_model_differences(t1: str, t2: str):

    # Similiarty output
    sim_output = {"string1": t1, "string2": t2}

    # Spacy Small
    v1 = nlp_small(t1).vector.tolist()
    v2 = nlp_small(t2).vector.tolist()
    spacy_small_sim = similarity(v1, v2)
    sim_output["spacy_small"] = spacy_small_sim

    # Spacy Medium
    v1 = nlp_medium(t1).vector.tolist()
    v2 = nlp_medium(t2).vector.tolist()
    spacy_medium_sim = similarity(v1, v2)
    sim_output["spacy_medium"] = spacy_medium_sim

    # Spacy Large
    v1 = nlp_lg(t1).vector.tolist()
    v2 = nlp_lg(t2).vector.tolist()
    spacy_large_sim = similarity(v1, v2)
    sim_output["spacy_large"] = spacy_large_sim

    # Sentence encode
    v1 = st_model.encode(t1).tolist()
    v2 = st_model.encode(t2).tolist()
    sentence_encoder_sim = similarity(v1, v2)
    sim_output["sentence_transformer"] = sentence_encoder_sim

    # Sentence transformer 768 encode
    v1 = st_768_model.encode(t1).tolist()
    v2 = st_768_model.encode(t2).tolist()
    sentence_encoder_sim = similarity(v1, v2)
    sim_output["sentence_transformer_768"] = sentence_encoder_sim

     # BERT encode
    v1 = bert_nlp(t1).tolist()
    v2 = bert_nlp(t2).tolist()
    bert_sim = similarity(v1, v2)
    sim_output["bert"] = bert_sim

    return sim_output
