"""Tratamentos aplicados sobre as features (decididos no notebook 02 de EDA)."""
import pandas as pd

from src.config import EMB_COLS


def remover_duplicatas(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove imagens repetidas (embeddings idênticos após arredondar em 4 casas).

    Mantém a primeira ocorrência. Determinístico: rodar duas vezes dá o mesmo resultado.
    """
    chave = pd.util.hash_pandas_object(df[EMB_COLS].round(4), index=False)
    manter = ~chave.duplicated(keep="first").to_numpy()
    if verbose:
        print(f"Duplicatas removidas: {(~manter).sum()} | base final: {manter.sum()} imagens")
    return df.loc[manter].reset_index(drop=True)


def carregar_base(caminho) -> pd.DataFrame:
    """Lê o parquet de features e aplica todos os tratamentos da EDA."""
    return remover_duplicatas(pd.read_parquet(caminho))
