# Plano de execução · VerIdade (AV2 Aprendizado de Máquina)

Entrega: **11/12/2026 às 23:59** (PDF do relatório + link do GitHub). Atraso ou formato errado = -20%.

## Princípios para ser eficiente

- **Embeddings uma vez só.** Tudo depois da Fase 1 lê `features_farl.parquet`. Nada de reprocessar imagem.
- **Splits congelados.** A coluna `split` é criada uma vez. Todo experimento usa ela (regra do professor).
- **`src/config.py` é a fonte única** de sementes, limiares e caminhos.
- **MLflow desde o primeiro modelo.** Não deixar para registrar no fim.
- **Infra cedo, mesmo vazia.** O `docker compose up` com Airflow tem que subir até 30/11, antes de ter modelo.
- **Relatório em paralelo.** Ao fechar cada fase, escrever a seção correspondente no template SBC.

## Fases

| Fase | Datas | Entregável | Bloco da nota |
|---|---|---|---|
| 0 · Setup | 23/11 | Repo no GitHub, Drive com datasets | |
| 1 · Dados | 24 a 26/11 | `features_farl.parquet` + notebook 02 EDA com 5+ gráficos | Análise 15% |
| 2 · Modelos clássicos | 27 a 29/11 | Baseline + KNN, Árvore, RF, AdaBoost, MLP; validação; Optuna; MLflow | Modelagem 30% |
| 2b · Esqueleto da infra | 29 e 30/11 | `docker compose up` com Mosquitto, Postgres, MLflow, Airflow subindo | MLOps 25% |
| 3 · Reprodução + contribuição | 01 a 03/12 | MultiAge (só idade vs multi-task), F2 @ recall 0,99, 3 sementes, cabeças 13/16, bootstrap | Modelagem 30% |
| 4 · Integração | 04 a 08/12 | Simulador → MQTT → Postgres → DAG → API → Streamlit | MLOps 25% |
| 5 · Fechamento | 09 a 11/12 | Relatório SBC (PDF), README, ensaio da demo, entrega | Comunicação 20% |

## Detalhe por fase

### Fase 1 · Dados
- Rodar `notebooks/01_extracao_embeddings_colab.ipynb` (GPU T4)
- Notebook 02 EDA: distribuição de idade, proporção de menores por limiar, gênero e raça por faixa (viés), qualidade das imagens, PCA/t-SNE dos embeddings colorido por idade, comparação UTKFace vs FG-NET (mudança de domínio)
- Justificar tratamentos: corte de idade > 101, recorte de rosto no FG-NET, normalização L2 dos embeddings

### Fase 2 · Modelos clássicos (sobre os embeddings)
- Tarefa principal: **menor de 18 vs adulto** (classificação) + **idade** (regressão)
- Baseline: Regressão Logística (e Linear para idade)
- Comparação: KNN, Árvore de Decisão, Random Forest, AdaBoost, MLP
- Validação: holdout (train/val/test), **stratified 5-fold** no treino, **leave-one-subject-out no FG-NET** (justifica o LOO: várias fotos por pessoa)
- Métricas: F2, recall de menores, taxa de falso adulto, ROC-AUC, MAE; matriz de confusão
- Optuna nos 2 melhores; análise de overfitting (curva treino vs validação)

### Fase 3 · Reprodução + contribuição
- MultiAge em PyTorch: MLP independente 512 → 256 → 512, cabeça de idade (102 classes) + cabeças binárias
- Reprodução: "só idade" vs "idade + cabeças 12/15/18/21"
- Ponto de operação: limiar ajustado na validação para recall de menores = 0,99; reportar F2
- Contribuição: + cabeças 13 e 16. Hipótese escrita **antes** de rodar
- 3 sementes (0, 1, 2) + IC 95% por bootstrap no teste e no FG-NET

### Fase 4 · Integração
- `embedded/`: simulador do totem publicando foto (base64) + id no tópico `veridade/captura`
- Ingestor: assina o tópico e grava no Postgres
- DAG Airflow (a cada 1 min): pega capturas pendentes → FaRL (CPU) → MultiAge → grava predição
- API FastAPI: `/predict` (foto avulsa), `/predicoes`, `/metricas`, `/health`
- Streamlit: fluxo ao vivo, % de menores bloqueados, distribuição de idade estimada, métricas do modelo (lidas do MLflow), status dos serviços

### Fase 5 · Fechamento
- Relatório SBC: Introdução (ECA Digital), Trabalhos relacionados, Dados, Metodologia, Resultados, Contribuição, Arquitetura MLOps, Ética e limitações, Conclusão
- Seção de ética: LGPD, ECA Digital, risco de viés por raça/gênero, impossibilidade de prova de idade só por rosto
