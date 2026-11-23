"""Configuração central do projeto.

Tudo que é "número mágico" (sementes, limiares, caminhos) mora aqui.
Notebooks, treino, DAGs e API importam daqui para garantir que todos
usem exatamente os mesmos valores.
"""
from pathlib import Path

# ---------------------------------------------------------------- caminhos
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_FILE = PROCESSED_DIR / "features_farl.parquet"

# ------------------------------------------------------------ reprodutibilidade
SEED_SPLIT = 42                 # semente fixa dos splits (nunca muda)
SEEDS_EXPERIMENTOS = [0, 1, 2]  # 3 sementes exigidas pelo professor

# Proporções do split do UTKFace (estratificado por faixa etária)
SPLIT_PROPORCOES = {"train": 0.70, "val": 0.15, "test": 0.15}

# ------------------------------------------------------------- problema
IDADE_MAX = 101                 # o artigo usa 102 classes (0 a 101)
LIMIARES_ARTIGO = [12, 15, 18, 21]          # reprodução
LIMIARES_CONTRIBUICAO = [12, 13, 15, 16, 18, 21]  # + limiares legais 13 e 16
LIMIAR_PRINCIPAL = 18

# Ponto de operação do artigo: recall de menores = 0,99
# (equivale a 1% de "falso adulto": menor classificado como adulto)
RECALL_ALVO = 0.99
F_BETA = 2  # F2 dá mais peso ao recall do que à precisão

# Faixas etárias usadas na estratificação e nos gráficos
FAIXAS_IDADE = [0, 3, 7, 10, 12, 13, 15, 16, 18, 21, 25, 30, 40, 50, 60, 70, 200]

# ------------------------------------------------------------- backbone
FARL_HF_ID = "Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64"
CLIP_PROCESSOR_ID = "openai/clip-vit-base-patch16"  # mesma normalização do CLIP ViT-B/16
EMB_DIM = 512
EMB_COLS = [f"e{i}" for i in range(EMB_DIM)]

# ------------------------------------------------------------- MLflow
MLFLOW_EXPERIMENTOS = {
    "baseline": "01-baseline",
    "comparacao": "02-comparacao-modelos",
    "tuning": "03-otimizacao-hiperparametros",
    "reproducao": "04-reproducao-multiage",
    "contribuicao": "05-contribuicao-limiares-legais",
}
