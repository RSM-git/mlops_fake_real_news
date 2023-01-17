# Base image
FROM python:3.9-slim

# install *some packages*
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY setup.py setup.py
COPY requirements.txt requirements.txt

# install requirements

# copy relevant folders
RUN pip install -r requirements.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu


COPY src/ src/
COPY data/processed data/processed
COPY data/raw data/raw
COPY models/ models/
COPY reports/ reports/
COPY .env .env
COPY configs/ configs/
COPY corded-pivot-374409-dbccae470422.json corded-pivot-374409-dbccae470422.json

# entrypoint
ENTRYPOINT ["python3", "-u", "src/models/train_model.py", "--config_file", "debug_cpu.yaml"]
