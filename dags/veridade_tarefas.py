"""Lógica das tarefas da DAG do VerIdade, sem dependência do Airflow (testável isoladamente).

Etapas: buscar pendentes -> validar qualidade -> inferir via API -> monitorar o modelo -> registrar no MLflow.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager

import math
import psycopg2
import psycopg2.extras
import requests

DB_URL = os.getenv("VERIDADE_DB_URL", "postgresql://veridade:veridade@localhost:5433/veridade")
API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
LOTE_MAX = int(os.getenv("LOTE_MAX", "1000"))
JANELA_MONITORAMENTO = int(os.getenv("JANELA_MONITORAMENTO", "300"))
ALERTA_FALSO_ADULTO = float(os.getenv("ALERTA_FALSO_ADULTO", "0.03"))  # 3x a meta de 1%
EXPERIMENTO_MLFLOW = "06-monitoramento-producao"


@contextmanager
def _conn():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


# ------------------------------------------------------------------ 1. ingestão
def buscar_pendentes() -> list[str]:
    """Capturas que chegaram pelo MQTT e ainda não foram processadas (mais antigas primeiro)."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT captura_id FROM capturas WHERE status = 'pendente' ORDER BY recebido_em LIMIT %s",
                    (LOTE_MAX,))
        ids = [r[0] for r in cur.fetchall()]
    print(f"{len(ids)} capturas pendentes")
    return ids


# ------------------------------------------------------------------ 2. preparação / qualidade
def validar(ids: list[str]) -> list[str]:
    """Descarta embeddings corrompidos: dimensão errada, valores inválidos ou norma fora de ~1."""
    if not ids:
        return []
    validos, invalidos = [], []
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT captura_id, embedding FROM capturas WHERE captura_id = ANY(%s)", (ids,))
        for cid, emb in cur.fetchall():
            # Python puro (sem numpy): a imagem oficial do Airflow não traz o numpy
            if emb is None or len(emb) != 512:
                invalidos.append((cid, f"dimensão {0 if emb is None else len(emb)}"))
            elif not all(x is not None and math.isfinite(x) for x in emb):
                invalidos.append((cid, "valores não finitos"))
            elif not 0.9 <= math.sqrt(sum(x * x for x in emb)) <= 1.1:
                invalidos.append((cid, f"norma {math.sqrt(sum(x * x for x in emb)):.3f}"))
            else:
                validos.append(cid)
        if invalidos:
            psycopg2.extras.execute_batch(
                cur, "UPDATE capturas SET status = 'invalida', motivo_invalida = %s WHERE captura_id = %s",
                [(motivo, cid) for cid, motivo in invalidos])
    print(f"{len(validos)} válidas, {len(invalidos)} inválidas")
    return validos


# ------------------------------------------------------------------ 3. inferência (disponibilização)
def inferir(ids: list[str], dag_run_id: str | None = None, tamanho_lote: int = 200) -> int:
    """Envia os embeddings para a API em lotes e grava as predições."""
    if not ids:
        return 0
    total = 0
    with _conn() as conn, conn.cursor() as cur:
        for i in range(0, len(ids), tamanho_lote):
            parte = ids[i:i + tamanho_lote]
            cur.execute("SELECT captura_id, embedding FROM capturas WHERE captura_id = ANY(%s)", (parte,))
            itens = [{"captura_id": cid, "embedding": [float(x) for x in emb]} for cid, emb in cur.fetchall()]
            resp = requests.post(f"{API_URL}/predict/lote", json={"itens": itens}, timeout=60)
            resp.raise_for_status()
            linhas = [(r["captura_id"], r["idade_estimada"], psycopg2.extras.Json(r["probabilidades"]),
                       psycopg2.extras.Json(r["decisoes"]), r["menor_18"], r["versao_modelo"], dag_run_id)
                      for r in resp.json()]
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO predicoes (captura_id, idade_estimada, probabilidades, decisoes, menor_18, versao_modelo, dag_run_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (captura_id) DO NOTHING""", linhas)
            cur.execute("UPDATE capturas SET status = 'processada' WHERE captura_id = ANY(%s)",
                        ([l[0] for l in linhas],))
            total += len(linhas)
    print(f"{total} predições gravadas")
    return total


# ------------------------------------------------------------------ 4. monitoramento
def monitorar(dag_run_id: str | None = None) -> dict:
    """Saúde do modelo nas últimas N predições com rótulo conhecido (disponível na simulação)."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            WITH janela AS (
                SELECT p.idade_estimada, p.menor_18, c.idade_real
                FROM predicoes p JOIN capturas c USING (captura_id)
                WHERE c.idade_real IS NOT NULL
                ORDER BY p.processado_em DESC, p.id DESC LIMIT %s)
            SELECT count(*), avg(abs(idade_estimada - idade_real)),
                   sum(CASE WHEN idade_real < 18 AND NOT menor_18 THEN 1 ELSE 0 END)::float
                     / NULLIF(sum(CASE WHEN idade_real < 18 THEN 1 ELSE 0 END), 0),
                   sum(CASE WHEN idade_real >= 18 AND menor_18 THEN 1 ELSE 0 END)::float
                     / NULLIF(sum(CASE WHEN idade_real >= 18 THEN 1 ELSE 0 END), 0),
                   avg(CASE WHEN menor_18 THEN 1.0 ELSE 0.0 END)::float
            FROM janela""", (JANELA_MONITORAMENTO,))
        n, mae, fa, fm, pct = cur.fetchone()
        if not n:
            print("Sem predições para monitorar ainda")
            return {}
        alerta = None
        if fa is not None and fa > ALERTA_FALSO_ADULTO and n >= 100:
            alerta = f"falso adulto {100 * fa:.1f}% acima do limite de {100 * ALERTA_FALSO_ADULTO:.0f}%"
        cur.execute("""INSERT INTO monitoramento (dag_run_id, n_janela, mae, taxa_falso_adulto, taxa_falso_menor,
                       pct_bloqueados_18, alerta) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (dag_run_id, n, mae, fa, fm, pct, alerta))
    m = {"n_janela": n, "mae": mae, "taxa_falso_adulto": fa, "taxa_falso_menor": fm, "pct_bloqueados_18": pct}
    print("Monitoramento:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
          "| ALERTA:" if alerta else "", alerta or "")
    return {k: float(v) if v is not None else None for k, v in m.items()} | {"alerta": alerta}


# ------------------------------------------------------------------ 5. registro no MLflow (API REST, sem dependências)
def _mlflow(rota: str, **corpo):
    r = requests.post(f"{MLFLOW_URL}/api/2.0/mlflow/{rota}", json=corpo, timeout=10)
    r.raise_for_status()
    return r.json()


def registrar_mlflow(metricas: dict, n_processadas: int, dag_run_id: str | None = None):
    if not metricas:
        return None
    try:
        r = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/get-by-name",
                         params={"experiment_name": EXPERIMENTO_MLFLOW}, timeout=10)
        exp_id = (r.json()["experiment"]["experiment_id"] if r.ok
                  else _mlflow("experiments/create", name=EXPERIMENTO_MLFLOW)["experiment_id"])
        agora = int(time.time() * 1000)
        run = _mlflow("runs/create", experiment_id=exp_id, start_time=agora, run_name=f"dag {dag_run_id or 'manual'}",
                      tags=[{"key": "origem", "value": "airflow"}, {"key": "dag_run_id", "value": str(dag_run_id)},
                            {"key": "alerta", "value": metricas.get("alerta") or "nenhum"}])["run"]["info"]["run_id"]
        valores = {k: v for k, v in metricas.items() if isinstance(v, float)} | {"processadas_nesta_execucao": float(n_processadas)}
        _mlflow("runs/log-batch", run_id=run,
                metrics=[{"key": k, "value": v, "timestamp": agora, "step": 0} for k, v in valores.items()])
        _mlflow("runs/update", run_id=run, status="FINISHED", end_time=int(time.time() * 1000))
        print(f"Registrado no MLflow (experimento {EXPERIMENTO_MLFLOW}, run {run})")
        return run
    except requests.RequestException as e:  # MLflow fora do ar não deve parar o pipeline
        print(f"Aviso: não foi possível registrar no MLflow ({e})")
        return None
