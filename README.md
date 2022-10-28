# VectorService
A small vector API service for generating Small, Medium and Large language vectors from Spacy using FastAPI

The service will let you generate vectors (96d or 300d) for each language model as well as perform comparisons using the built in cosine similarity functions.

SimDiff endpoint will allow you to compare the outputs of all 3 models when comparing 2 strings of text.  This is useful for showing the difference in accuracy between these models.

## Installation

```
pip install spacy
pip install "fastapi[all]"
pip install sklearn

python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

## Running

```
uvicorn main:app --reload
```
