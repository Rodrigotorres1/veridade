-- Roda uma única vez, na primeira inicialização do container do Postgres.

-- Banco separado para os metadados do Airflow
CREATE DATABASE airflow OWNER veridade;

-- \connect garante que as tabelas abaixo fiquem no banco "veridade"
\connect veridade

-- Capturas recebidas dos totens via MQTT.
-- Por privacidade (LGPD / ECA Digital), o totem NÃO envia a foto: envia só o embedding
-- de 512 números calculado no próprio dispositivo. Não é possível reconstruir o rosto a partir dele.
CREATE TABLE IF NOT EXISTS capturas (
    id            BIGSERIAL PRIMARY KEY,
    captura_id    TEXT UNIQUE NOT NULL,
    totem_id      TEXT NOT NULL,
    capturado_em  TIMESTAMPTZ NOT NULL,
    recebido_em   TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding     REAL[] NOT NULL,
    idade_real    INTEGER,          -- só existe na simulação (rótulo do dataset), usado para monitorar o modelo
    origem        TEXT,             -- dataset de origem (UTKFace teste ou FG-NET)
    status        TEXT NOT NULL DEFAULT 'pendente',   -- pendente | processada | invalida
    motivo_invalida TEXT
);
CREATE INDEX IF NOT EXISTS idx_capturas_status ON capturas (status, recebido_em);

-- Resultado da inferência feita pela DAG do Airflow (via API)
CREATE TABLE IF NOT EXISTS predicoes (
    id             BIGSERIAL PRIMARY KEY,
    captura_id     TEXT UNIQUE NOT NULL REFERENCES capturas (captura_id),
    processado_em  TIMESTAMPTZ NOT NULL DEFAULT now(),
    idade_estimada REAL NOT NULL,
    probabilidades JSONB NOT NULL,  -- {"12": 0.01, "13": 0.02, ...} = P(menor que o limiar)
    decisoes       JSONB NOT NULL,  -- {"12": false, ..., "18": true} = bloqueado como menor?
    menor_18       BOOLEAN NOT NULL,
    versao_modelo  TEXT NOT NULL,
    dag_run_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_predicoes_tempo ON predicoes (processado_em);

-- Uma linha por execução da DAG: saúde do modelo em produção
CREATE TABLE IF NOT EXISTS monitoramento (
    id                 BIGSERIAL PRIMARY KEY,
    registrado_em      TIMESTAMPTZ NOT NULL DEFAULT now(),
    dag_run_id         TEXT,
    n_janela           INTEGER NOT NULL,
    mae                REAL,
    taxa_falso_adulto  REAL,
    taxa_falso_menor   REAL,
    pct_bloqueados_18  REAL,
    alerta             TEXT
);
