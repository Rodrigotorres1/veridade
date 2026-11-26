"""API de inferência do VerIdade (FastAPI).

Carrega o MultiAge final (models/multiage) e devolve, para cada embedding:
  - idade estimada
  - P(menor que T) para cada limiar legal (12, 13, 15, 16, 18, 21)
  - decisão por limiar, usando o limiar de operação calibrado na validação (recall de menores = 0,99)

Também expõe as predições gravadas e as métricas de monitoramento para o dashboard.
Documentação interativa: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.multiage import MultiAge  # noqa: E402

PASTA_MODELO = Path(os.getenv("PASTA_MODELO", Path(__file__).resolve().parents[1] / "models" / "multiage"))
DB_URL = os.getenv("VERIDADE_DB_URL", "postgresql://veridade:veridade@localhost:5433/veridade")
ESTADO: dict = {}


def carregar_modelo():
    cfg = json.loads((PASTA_MODELO / "config.json").read_text(encoding="utf-8"))
    modelo = MultiAge(cfg["cabecas"])
    modelo.load_state_dict(torch.load(PASTA_MODELO / "modelo.pt", map_location="cpu"))
    modelo.eval()
    versao = f"multiage-{cfg['configuracao'].split(' ')[0]}-seed{cfg['seed']}"
    return modelo, cfg, versao


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    ESTADO["modelo"], ESTADO["config"], ESTADO["versao"] = carregar_modelo()
    print(f"[api] modelo carregado: {ESTADO['versao']} | cabeças {ESTADO['config']['cabecas']}", flush=True)
    yield


app = FastAPI(title="VerIdade API", version="1.0",
              description="Triagem de menores de idade a partir de embeddings faciais (FaRL + MultiAge).",
              lifespan=ciclo_de_vida)


# ------------------------------------------------------------------ esquemas
class Entrada(BaseModel):
    embedding: list[float] = Field(..., min_length=512, max_length=512, description="Embedding FaRL normalizado (512 valores)")


class ItemLote(Entrada):
    captura_id: str


class Lote(BaseModel):
    itens: list[ItemLote] = Field(..., max_length=1000)


def _inferir(embeddings: np.ndarray) -> list[dict]:
    z = torch.tensor(embeddings, dtype=torch.float32)
    saida = ESTADO["modelo"].prever(z)
    limiares_op = ESTADO["config"]["limiares_operacao"]
    resultados = []
    for i in range(len(z)):
        probs = {t: round(float(saida[f"p_menor_{t}"][i]), 5) for t in limiares_op}
        decisoes = {t: probs[t] >= limiares_op[t] for t in limiares_op}
        resultados.append({
            "idade_estimada": round(float(saida["idade"][i]), 2),
            "probabilidades": probs,
            "decisoes": decisoes,
            "menor_18": decisoes["18"],
            "versao_modelo": ESTADO["versao"],
        })
    return resultados


def _validar(m: np.ndarray):
    if not np.isfinite(m).all():
        raise HTTPException(422, "Embedding contém valores inválidos (NaN ou infinito)")
    normas = np.linalg.norm(m, axis=1)
    if ((normas < 0.9) | (normas > 1.1)).any():
        raise HTTPException(422, "Embedding fora do padrão: a norma deveria ser ~1 (saída normalizada do FaRL)")


# ------------------------------------------------------------------ inferência
@app.get("/health")
def health():
    db_ok = True
    try:
        psycopg2.connect(DB_URL, connect_timeout=2).close()
    except Exception:
        db_ok = False
    return {"status": "ok", "modelo": ESTADO.get("versao"), "banco": "ok" if db_ok else "indisponível"}


@app.get("/modelo")
def modelo_info():
    return {"versao": ESTADO["versao"], **ESTADO["config"],
            "descricao": "FaRL ViT-B/16 congelado + MultiAge I-256 (Gaul et al., 2025) com cabeças nos limiares legais 13 e 16"}


@app.post("/predict")
def predict(entrada: Entrada):
    m = np.asarray([entrada.embedding], dtype=np.float32); _validar(m)
    return _inferir(m)[0]


@app.post("/predict/lote")
def predict_lote(lote: Lote):
    if not lote.itens:
        return []
    m = np.asarray([it.embedding for it in lote.itens], dtype=np.float32); _validar(m)
    return [{"captura_id": it.captura_id, **r} for it, r in zip(lote.itens, _inferir(m))]


# ------------------------------------------------------------------ leitura para o dashboard
def _consulta(sql: str, params=None) -> list[dict]:
    try:
        with psycopg2.connect(DB_URL, connect_timeout=3) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    except psycopg2.Error as e:
        raise HTTPException(503, f"Banco indisponível: {e}")


@app.get("/predicoes")
def predicoes(limite: int = 50):
    return _consulta("""
        SELECT p.captura_id, c.totem_id, c.origem, c.capturado_em, p.processado_em,
               p.idade_estimada, p.menor_18, p.decisoes, c.idade_real
        FROM predicoes p JOIN capturas c USING (captura_id)
        ORDER BY p.processado_em DESC, p.id DESC LIMIT %s""", (min(limite, 500),))


@app.get("/metricas")
def metricas():
    status = {r["status"]: r["n"] for r in _consulta("SELECT status, count(*) AS n FROM capturas GROUP BY status")}
    geral = _consulta("""
        SELECT count(*) AS processadas,
               avg(CASE WHEN p.menor_18 THEN 1.0 ELSE 0.0 END)::float AS pct_bloqueados_18,
               avg(abs(p.idade_estimada - c.idade_real)) AS mae,
               sum(CASE WHEN c.idade_real < 18 AND NOT p.menor_18 THEN 1 ELSE 0 END)::float
                 / NULLIF(sum(CASE WHEN c.idade_real < 18 THEN 1 ELSE 0 END), 0) AS taxa_falso_adulto,
               sum(CASE WHEN c.idade_real >= 18 AND p.menor_18 THEN 1 ELSE 0 END)::float
                 / NULLIF(sum(CASE WHEN c.idade_real >= 18 THEN 1 ELSE 0 END), 0) AS taxa_falso_menor
        FROM predicoes p JOIN capturas c USING (captura_id)""")[0]
    por_limiar = _consulta("""
        SELECT chave AS limiar, avg(CASE WHEN valor::boolean THEN 1.0 ELSE 0.0 END)::float AS pct_bloqueados
        FROM predicoes, jsonb_each_text(decisoes) AS d(chave, valor)
        GROUP BY chave ORDER BY chave::int""")
    por_minuto = _consulta("""
        SELECT date_trunc('minute', recebido_em) AS minuto, count(*) AS capturas
        FROM capturas WHERE recebido_em > now() - interval '30 minutes'
        GROUP BY 1 ORDER BY 1""")
    por_totem = _consulta("""
        SELECT c.totem_id, count(*) AS processadas, avg(CASE WHEN p.menor_18 THEN 1.0 ELSE 0.0 END)::float AS pct_bloqueados_18
        FROM predicoes p JOIN capturas c USING (captura_id) GROUP BY 1 ORDER BY 1""")
    monit = _consulta("SELECT * FROM monitoramento ORDER BY registrado_em DESC LIMIT 60")
    return {"capturas": {"recebidas": sum(status.values()), "pendentes": status.get("pendente", 0),
                         "processadas": status.get("processada", 0), "invalidas": status.get("invalida", 0)},
            "geral": geral, "por_limiar": por_limiar, "por_minuto": por_minuto,
            "por_totem": por_totem, "monitoramento": monit}
