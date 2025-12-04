FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt