# Base image
FROM python:3.9-slim

# install *some packages*
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY setup.py setup.py
COPY requirements.txt requirements.txt

# install requirements
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# copy relevant folders
COPY src/ src/

# should these be included?
COPY tests/ tests/
COPY tokenizers/ tokenizers/

# SECRET FILES
COPY .env .env
COPY application_default_credentials.json auth.json

# uvicorn - start hosting
CMD exec uvicorn src.models.predict_model:app --port 8000 --host 0.0.0.0 --workers 1
