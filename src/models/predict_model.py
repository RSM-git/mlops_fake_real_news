import os

from fastapi import FastAPI
from google.cloud import storage

from src.models.model import FakeNewsClassifier

app = FastAPI()

# gcloud bucket details
bucket_name = "fake_real_news_bucket"
blob_name = "best_model.ckpt"

# get model
storage_client = storage.Client(project="corded-pivot-374409")
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(blob_name)

# store model
file_name = "src/models/" + blob_name
blob.download_to_filename(file_name)

# initialize model
model = FakeNewsClassifier.load_from_checkpoint(os.getcwd() + file_name)
model.eval()


@app.get("/")
async def root():
    return {"Root": "Root"}


# enable two different ways of requesting from the api
@app.get("/predict/{text}")
async def predict_get(text: str):
    prediction = model.predict_from_str(text)
    return {"prediction": prediction}


@app.post("/predict/")
async def predict_post(text: str):
    prediction = model.predict_from_str(text)
    return prediction
