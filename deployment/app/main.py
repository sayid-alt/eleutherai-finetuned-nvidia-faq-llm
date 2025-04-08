from transformers import pipeline
from fastapi import FastAPI

# define model info
model_name = "paacamo/EleutherAI-pythia-1b-finetuned-nvidia-faq"
pipe = pipeline('text-generation', model=model_name, device=0)

# Create FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}