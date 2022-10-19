# Totally minimal example with the large web model for use on your laptop

from fastapi import FastAPI
import spacy

nlp_lg = spacy.load('en_core_web_lg')

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Feed me text and I pop out vector clouds. See /docs for more info."}

@app.get("/vec/")
async def vectorize_text_large(text: str):
    doc = nlp_lg(text)
    return doc.vector.tolist()

@app.get("/sim/")
async def similarity_text_large(t1: str, t2: str):
    d1 = nlp_lg(t1)
    d2 = nlp_lg(t2)
    return d1.similarity(d2)
