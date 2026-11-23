# VerIdade

> Triagem de menores de idade em imagens faciais, alinhada ao ECA Digital (Lei 15.211/2025).
> Reprodução em escala reduzida de *"Underage Detection through a Multi-Task and MultiAge Approach for Screening Minors in Unconstrained Imagery"* (Gaul et al., 2025), com solução completa de MLOps.

**Disciplina:** Aprendizado de Máquina · CESAR School · 2026.2 · Turma B

## Integrante

| Nome | GitHub |
|---|---|
| Rodrigo Torres | @SEU_USUARIO |

**Google Sites do projeto:** _(link)_

## Descrição

O VerIdade simula um totem de verificação de idade. Um sistema embarcado (simulado) captura fotos de rosto e publica via MQTT. O Airflow processa as imagens novas, extrai embeddings com o FaRL congelado e o modelo **MultiAge** estima a idade e a probabilidade de a pessoa estar abaixo de 12, 13, 15, 16, 18 e 21 anos. As inferências ficam disponíveis por API e são acompanhadas em tempo real num dashboard.

**Contribuição:** hipótese de que cabeças binárias nos limiares legais (13 e 16 anos) melhoram a detecção nesses cortes sem piorar o limiar de 18 anos. Avaliada com os mesmos splits, 3 sementes e intervalo de confiança por bootstrap.

## Arquitetura

```
[embedded/ simulador] --MQTT--> [Mosquitto] --> [ingestor] --> [PostgreSQL]
                                                                   |
                         [Airflow: DAG ingestão -> FaRL -> inferência]
                                                                   |
                    [MLflow: experimentos e modelo]  -->  [api/ FastAPI]  -->  [app/ Streamlit]
```

## Como executar

### 1. Gerar as features (uma vez, no Google Colab)

As imagens não são redistribuídas (licença do UTKFace: uso não comercial em pesquisa).

1. Baixe o **UTKFace Aligned & Cropped** (`UTKFace.tar.gz`) e o **FG-NET** (`FGNET.zip`)
2. Coloque em `MyDrive/projeto-ml/raw/` no Google Drive
3. Rode `notebooks/01_extracao_embeddings_colab.ipynb` com GPU
4. Copie `features_farl.parquet` para `data/processed/`

### 2. Subir toda a solução

```bash
docker compose up --build
```

| Serviço | Endereço |
|---|---|
| Dashboard | http://localhost:8501 |
| API (docs) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

## Estrutura

```
data/        dados (raw fora do git; processed = embeddings)
notebooks/   01 extração (Colab) · 02 EDA · 03 modelagem
src/         config, dados, modelos, treino
embedded/    simulador do totem (publica no MQTT)
dags/        DAGs do Airflow
api/         API de inferência (FastAPI)
app/         dashboard (Streamlit)
mlruns/      MLflow
reports/     relatório em PDF (template SBC)
```

## Considerações éticas

O projeto lida com rostos de menores. Nenhuma imagem é versionada ou exibida no dashboard; apenas resultados agregados e previsões. A discussão sobre LGPD, ECA Digital e os limites da estimativa facial de idade está no relatório.
