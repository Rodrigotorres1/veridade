"""Configuração do MLflow.

Local/Colab: banco SQLite em mlruns/mlflow.db e artefatos em mlruns/artifacts.
Docker: a variável MLFLOW_TRACKING_URI aponta para o servidor (http://mlflow:5000).
"""
import os

import mlflow

from src.config import ROOT


def configurar_mlflow() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        (ROOT / "mlruns" / "artifacts").mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    return uri


def usar_experimento(nome: str) -> str:
    exp = mlflow.get_experiment_by_name(nome)
    if exp is None:
        artefatos = None if os.getenv("MLFLOW_TRACKING_URI") else (ROOT / "mlruns" / "artifacts" / nome).as_uri()
        exp_id = mlflow.create_experiment(nome, artifact_location=artefatos)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id
