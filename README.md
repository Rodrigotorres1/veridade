# VerIdade

> Triagem de menores de idade em verificação facial, alinhada ao ECA Digital (Lei 15.211/2025).
> Reprodução em escala reduzida de *"Underage Detection through a Multi-Task and MultiAge Approach for Screening Minors in Unconstrained Imagery"* (Gaul et al., 2025), com contribuição própria e solução completa de MLOps.

**Disciplina:** Aprendizado de Máquina · CESAR School · 2026.2 · Turma B

## Integrante

| Nome | GitHub |
|---|---|
| Rodrigo Torres | [@Rodrigotorres1](https://github.com/Rodrigotorres1) |

## Descrição

O ECA Digital proíbe a autodeclaração de idade, o que coloca modelos de estimativa facial de idade no centro do debate. O VerIdade simula uma rede de **totens de verificação de idade**: cada totem calcula no próprio dispositivo o embedding facial (FaRL) e publica **apenas esse vetor** via MQTT, nunca a foto. O Airflow processa as capturas a cada minuto, chama a API com o modelo **MultiAge** (idade estimada + decisão em 6 limiares legais) e monitora o desempenho, que é acompanhado em tempo real num dashboard.

**Contribuição:** cabeças binárias dedicadas aos limiares legais de **13 anos** (cadastro em redes sociais) e **16 anos** (consentimento). Resultado: ganho significativo em 16 anos, sem piora em 18; resultado negativo em 13 anos, analisado no relatório.

## Arquitetura

```
 ┌──────────────┐  MQTT (QoS 1)   ┌───────────┐     ┌──────────┐     ┌────────────┐
 │ totem        │ ──────────────▶ │ mosquitto │ ──▶ │ ingestor │ ──▶ │ PostgreSQL │
 │ (embarcado   │ veridade/totem/ └───────────┘     └──────────┘     └─────┬──────┘
 │  simulado)   │ <id>/captura                                              │
 └──────────────┘                                                           │
                    ┌──────────────── Airflow (a cada 1 min) ───────────────┘
                    │ buscar pendentes → validar → inferir → monitorar → registrar
                    ▼                                │                       │
              ┌───────────┐                          ▼                       ▼
              │ API       │ ◀── predições     ┌────────────┐          ┌──────────┐
              │ FastAPI + │ ──────────────▶   │ PostgreSQL │          │ MLflow   │
              │ MultiAge  │                   └────────────┘          └──────────┘
              └─────┬─────┘
                    ▼
              ┌───────────┐
              │ Dashboard │  Streamlit, atualização automática
              └───────────┘
```

| Serviço | Pasta | Função |
|---|---|---|
| `totem` | `embedded/totem.py` | Sistema embarcado simulado: publica embeddings de imagens nunca vistas no treino |
| `mosquitto` | `docker/mosquitto` | Broker MQTT |
| `ingestor` | `embedded/ingestor.py` | Assina o MQTT e grava as capturas como "pendentes" |
| `postgres` | `docker/postgres` | Capturas, predições e monitoramento (+ metadados do Airflow) |
| `airflow` | `dags/` | DAG `veridade_pipeline`: validação, inferência, monitoramento e registro no MLflow |
| `api` | `api/` | FastAPI com o MultiAge final (`models/multiage`) |
| `mlflow` | `docker/mlflow` | Experimentos dos notebooks + monitoramento de produção |
| `dashboard` | `app/` | Streamlit com métricas, decisões por limiar e saúde do modelo |

## Como executar

### Pré-requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e aberto
- O arquivo `data/processed/features_farl.parquet` (gerado pelo notebook 01; não vai para o GitHub por causa da licença do UTKFace)

### 1. Gerar as features (uma vez, no Google Colab)
1. Baixe o **UTKFace Aligned & Cropped** (`UTKFace.tar.gz`) e o **FG-NET** (`FGNET.zip`) e coloque em `MyDrive/projeto-ml/raw/`
2. Rode `notebooks/01_extracao_embeddings_colab.ipynb` com GPU
3. Copie `features_farl.parquet` para `data/processed/`

### 2. Subir toda a solução
```bash
docker compose up --build
```
A primeira vez leva alguns minutos (download das imagens e do PyTorch). Depois:

| Serviço | Endereço | Login |
|---|---|---|
| Dashboard | http://localhost:8501 | |
| API (docs interativas) | http://localhost:8000/docs | |
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5000 | |

O totem começa a publicar sozinho. Em até 1 minuto a DAG roda e as predições aparecem no dashboard.

### 3. (Opcional) Ver os experimentos dos notebooks no MLflow
Copie o conteúdo de `MyDrive/projeto-ml/resultados/mlruns/` para a pasta `mlruns/` do repositório antes do `docker compose up`.

### Parar
```bash
docker compose down        # mantém o banco
docker compose down -v     # apaga também o banco
```

## Notebooks

| Notebook | Conteúdo |
|---|---|
| `01_extracao_embeddings_colab` | Extração dos embeddings FaRL (Colab, GPU) e splits congelados |
| `02_eda` | Análise exploratória: 8 figuras, qualidade dos dados e decisões de tratamento |
| `03_modelagem_classica` | Baseline + KNN, Árvore, RF, AdaBoost e MLP; k-fold, LOO, overfitting, Grid Search e Optuna |
| `04_reproducao_contribuicao` | MultiAge: só idade × multi-task × multi-task + 13/16 anos, 3 sementes e bootstrap |

## Estrutura

```
api/         API de inferência (FastAPI)
app/         dashboard (Streamlit)
dags/        DAG do Airflow (lógica em veridade_tarefas.py)
data/        dados (raw fora do git; processed = embeddings)
docker/      configuração do Mosquitto, Postgres e MLflow
embedded/    totem simulado e ingestor MQTT
mlruns/      MLflow
models/      modelo final (MultiAge C, seed 0)
notebooks/   01 a 04
reports/     figuras e relatório em PDF (template SBC)
src/         código compartilhado: config, dados, modelos, treino e avaliação
```

## Considerações éticas

- **Privacidade desde a concepção:** os totens enviam só o embedding; nenhuma imagem é transmitida, armazenada ou exibida
- **Viés:** a taxa de falso adulto variou entre grupos raciais no teste (0% a 4,35%), com amostras pequenas; discutido no relatório
- **Limites:** a estimativa facial de idade não é prova de idade; o sistema é uma triagem que encaminha casos duvidosos para verificação adicional
- **Licença dos dados:** UTKFace e FG-NET são de uso não comercial em pesquisa e não são redistribuídos aqui
