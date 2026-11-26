FROM python:3.11-slim
WORKDIR /app
# PyTorch só de CPU (bem menor que a versão com CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "psycopg2-binary>=2.9" "numpy>=1.26"
COPY src/__init__.py src/config.py src/
COPY src/models/ src/models/
COPY api/ api/
ENV PYTHONUNBUFFERED=1 PASTA_MODELO=/app/models/multiage
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
