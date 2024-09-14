FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt 

EXPOSE 8000

CMD ["uvicorn", "app:app", "--port", "8000", "--host", "0.0.0.0"]