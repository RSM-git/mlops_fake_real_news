import os

from fastapi import FastAPI
from google.cloud import storage

from src.models.model import FakeNewsClassifier

app = FastAPI()

bucket_name = "fake_real_news_bucket"
blob_name = "best_model.ckpt"

storage_client = storage.Client(project="corded-pivot-374409")
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(blob_name)

file_name = "src/models/" + blob_name
blob.download_to_filename(file_name)

model = FakeNewsClassifier.load_from_checkpoint(os.getcwd() + file_name)
model.eval()


@app.get("/")
async def root():
    return {"Root": "Root"}


@app.get("/predict/{text}")
async def predict_get(text: str):
    prediction = model.predict_from_str(text)
    return {"prediction": prediction}


@app.post("/predict/")
async def predict_post(text: str):
    prediction = model.predict_from_str(text)
    return prediction
