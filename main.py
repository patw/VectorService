from fastapi import FastAPI
import spacy
from sklearn.preprocessing import normalize
import numpy

# Load the spacy model that you have installed
nlp_small = spacy.load('en_core_web_sm')
nlp_medium = spacy.load('en_core_web_md')
nlp_lg = spacy.load('en_core_web_lg')

# Fast API init
app = FastAPI(
        title="TextVectorizor",
        description="Create vector clouds using the Spacy English language small (96d), medium (300d) and large (300d) models. Use the similarity tools to compare words and phrases.",
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

# L2 normalization on vectors using sklearn utility function
def vector_normalize(vec):
    shaped = vec.reshape(-1,1)
    normed = normalize(shaped,axis=0)
    return normed.reshape(1,-1)[0]

# Most Similar Word Function (stolen from Spacy github issues!)
def most_similar_words(nlp, word, num_words):
    ms = nlp.vocab.vectors.most_similar(numpy.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]), n=num_words)
    return [nlp.vocab.strings[w] for w in ms[0][0]]

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

@app.get("/simdiff/")
async def similarity_model_differences(t1: str, t2: str):
    # Small Model
    d1 = nlp_small(t1)
    d2 = nlp_small(t2)
    s1 = d1.similarity(d2)

    # Medium Model
    d1 = nlp_medium(t1)
    d2 = nlp_medium(t2)
    s2 = d1.similarity(d2)

    # Large Model
    d1 = nlp_lg(t1)
    d2 = nlp_lg(t2)
    s3 = d1.similarity(d2)

    return {'small': s1, 'medium': s2, 'large': s3}
