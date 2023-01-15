import os

from dotenv import load_dotenv
from fastapi import FastAPI
from google.cloud import storage

from src.models.model import FakeNewsClassifier

load_dotenv(".env")

app = FastAPI()

bucket_name = "model-bucket-test123"
blob_name = "best_model.ckpt"

storage_client = storage.Client(project="corded-pivot-374409")
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(blob_name)

file_name = "src/models/" + blob_name
blob.download_to_filename(file_name)

model = FakeNewsClassifier.load_from_checkpoint(os.getcwd() + file_name)
model.eval()


@app.get("/")
def root():
    return {"Hello": "World"}


@app.get("/predict/{text}")
def predict(text: str):
    prediction = model.predict_from_str(text)
    return {"prediction": prediction}
