FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY rag_eval ./rag_eval
COPY db ./db
COPY data ./data
COPY reports ./reports
EXPOSE 8000
# Needs Postgres (see docker-compose.yml). Offline by default; set LLM=openai and OPENAI_API_KEY for model answers.
CMD ["python", "-m", "rag_eval", "serve", "--host", "0.0.0.0", "--port", "8000"]
