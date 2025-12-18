FROM python:3.11-slim

WORKDIR /app

# Install docker CLI
RUN apt-get update && \
    apt-get install -y docker.io && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code_execution_server.py .

# Create artifacts directory
RUN mkdir -p /app/generated_artifacts

EXPOSE 8000

CMD ["uvicorn", "code_execution_server:app", "--host", "0.0.0.0", "--port", "8000"]
