FROM python:3.11-slim

# System deps: tesseract + poppler (pdf -> images) + basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Render usa PORT automáticamente
ENV PORT=10000
EXPOSE 10000

CMD ["bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
