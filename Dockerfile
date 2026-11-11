FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY rag_eval ./rag_eval
COPY data ./data
EXPOSE 8000
# Offline by default (no API key needed). Set EMBEDDINGS=openai GENERATOR=openai OPENAI_API_KEY=... for real models.
CMD ["python", "-m", "rag_eval", "serve", "--port", "8000"]
