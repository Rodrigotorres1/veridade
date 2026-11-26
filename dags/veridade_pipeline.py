"""DAG do VerIdade: processa as capturas recebidas dos totens a cada 1 minuto.

  buscar_pendentes -> validar -> inferir (API MultiAge) -> monitorar -> registrar_mlflow

A lógica de cada tarefa está em veridade_tarefas.py (testável sem o Airflow).
"""
from datetime import datetime, timedelta

from airflow.decorators import dag, task

import veridade_tarefas as t


@dag(
    dag_id="veridade_pipeline",
    description="Ingestão, validação, inferência e monitoramento das capturas dos totens",
    schedule=timedelta(minutes=1),
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=15)},
    tags=["veridade", "mlops"],
)
def veridade_pipeline():
    @task
    def buscar_pendentes() -> list[str]:
        return t.buscar_pendentes()

    @task
    def validar(ids: list[str]) -> list[str]:
        return t.validar(ids)

    @task
    def inferir(ids: list[str], run_id=None) -> int:
        return t.inferir(ids, dag_run_id=run_id)

    @task
    def monitorar(n_processadas: int, run_id=None) -> dict:
        return t.monitorar(dag_run_id=run_id)

    @task
    def registrar_mlflow(metricas: dict, n_processadas: int, run_id=None):
        return t.registrar_mlflow(metricas, n_processadas, dag_run_id=run_id)

    ids = buscar_pendentes()
    validos = validar(ids)
    n = inferir(validos)
    m = monitorar(n)
    registrar_mlflow(m, n)


veridade_pipeline()
