"""Splits congelados: gerados uma vez e salvos junto das features.

Todos os experimentos (baseline, comparação, reprodução, contribuição)
usam a MESMA coluna `split`, como pedem as regras da contribuição.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import FAIXAS_IDADE, SEED_SPLIT, SPLIT_PROPORCOES


def faixa_etaria(idades: pd.Series) -> pd.Series:
    return pd.cut(idades, bins=FAIXAS_IDADE, right=False).astype(str)


def criar_splits(df: pd.DataFrame) -> pd.DataFrame:
    """UTKFace -> train/val/test estratificado por faixa etária.
    FG-NET  -> test_externo (nunca entra no treino; mede generalização)."""
    df = df.copy()
    df["split"] = np.where(df["dataset"] == "FG-NET", "test_externo", None)

    utk = df[df["dataset"] == "UTKFace"]
    estratos = faixa_etaria(utk["idade"])
    p = SPLIT_PROPORCOES
    idx_train, idx_resto = train_test_split(
        utk.index, train_size=p["train"], stratify=estratos, random_state=SEED_SPLIT)
    idx_val, idx_test = train_test_split(
        idx_resto, test_size=p["test"] / (p["val"] + p["test"]),
        stratify=estratos.loc[idx_resto], random_state=SEED_SPLIT)

    df.loc[idx_train, "split"] = "train"
    df.loc[idx_val, "split"] = "val"
    df.loc[idx_test, "split"] = "test"
    return df
