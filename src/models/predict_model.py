import os

from src.models.model import FakeNewsClassifier

model = FakeNewsClassifier.load_from_checkpoint(os.getcwd() + "/models/best_model.ckpt")

# disable randomness, dropout, etc...
model.eval()

# predict with the model
y_hat = model.predict_from_str("Trump on Twitter (Dec 28) - Global Warming")
print(y_hat)
