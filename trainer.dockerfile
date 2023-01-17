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
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# copy relevant folders
COPY configs/ configs/
COPY src/ src/
COPY data/ data/
COPY models/ models/

RUN dvc pull

# entrypoint
ENTRYPOINT ["python3", "-u", "src/models/train_model.py"]
