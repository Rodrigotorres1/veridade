FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow>=3.5,<4"
WORKDIR /mlflow
EXPOSE 5000
# "db upgrade" atualiza o banco criado no Colab caso ele seja de uma versão anterior do MLflow
CMD mlflow db upgrade sqlite:////mlflow/mlflow.db 2>/dev/null; \
    mlflow server --host 0.0.0.0 --port 5000 \
      --backend-store-uri sqlite:////mlflow/mlflow.db \
      --artifacts-destination /mlflow/artifacts \
      --allowed-hosts "mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000"
