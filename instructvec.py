# A realistic, minimal Instructor large vectorizer using quantization to speed it up

from fastapi import FastAPI
from InstructorEmbedding import INSTRUCTOR
import torch

# Instructor model
model = INSTRUCTOR('hkunlp/instructor-large', device='cpu')
qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

app = FastAPI(title="InstructVectorizer",
        description="Text embedding quantized Instructor-large.",
        version="1.0",
        contact={
            "name": "Pat Wendorf",
            "email": "pat.wendorf@mongodb.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/license/mit/",
    })

@app.get("/")
async def root():
    return {"message": "Vectorize text with the Instructor-large model. See /docs for more info."}

@app.get("/vectorize/")
async def vectorize(text: str, instruction: str = "Represent the text for retrieval:"):
    embedding = qmodel.encode([[instruction,text]]).tolist()[0]
    return embedding
