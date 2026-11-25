"""Métricas e gráficos de avaliação, compartilhados entre os notebooks 03 e 04.

Convenção: classe positiva (1) = MENOR de idade.
"Falso adulto" = menor classificado como adulto (o erro mais grave no contexto do ECA Digital).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src.config import F_BETA, RECALL_ALVO


def limiar_para_recall(y_true, score, recall_alvo: float = RECALL_ALVO) -> float:
    """Maior limiar de decisão que ainda garante recall >= recall_alvo.

    Ajustado SEMPRE no conjunto de validação e depois aplicado, fixo, no teste.
    É o "ponto de operação" do artigo: no máximo 1% de falso adulto.
    """
    _, recall, limiares = precision_recall_curve(y_true, score)
    # recall tem len(limiares)+1 pontos; recall[i] corresponde a limiares[i]
    ok = np.where(recall[:-1] >= recall_alvo)[0]
    return float(limiares[ok[-1]]) if len(ok) else float(limiares[0])


def metricas_binarias(y_true, score, limiar: float, prefixo: str = "") -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(score) >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    m = {
        "f2": fbeta_score(y_true, pred, beta=F_BETA, zero_division=0),
        "recall_menor": recall_score(y_true, pred, zero_division=0),
        "precisao_menor": precision_score(y_true, pred, zero_division=0),
        "taxa_falso_adulto": fn / max(fn + tp, 1),
        "taxa_falso_menor": fp / max(fp + tn, 1),
        "roc_auc": roc_auc_score(y_true, score) if len(set(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, score) if len(set(y_true)) > 1 else np.nan,
    }
    return {f"{prefixo}{k}": float(v) for k, v in m.items()}


def taxa_falso_adulto_por_grupo(y_true, score, limiar, grupos) -> dict:
    """Fairness: falso adulto separado por grupo (ex.: raça). Só conta quem é menor."""
    y_true, pred, grupos = np.asarray(y_true), np.asarray(score) >= limiar, np.asarray(grupos)
    out = {}
    for g in np.unique(grupos[~pd_isna(grupos)]):
        m = (grupos == g) & (y_true == 1)
        if m.sum() >= 10:
            out[str(g)] = {"n_menores": int(m.sum()), "taxa_falso_adulto": float((~pred[m]).mean())}
    return out


def pd_isna(a):
    return np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in a])


# ------------------------------------------------------------------ gráficos
def plot_matriz_confusao(y_true, score, limiar, titulo, ax=None):
    ax = ax or plt.subplots(figsize=(4, 3.6))[1]
    cm = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(score) >= limiar).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    rot = ["Adulto", "Menor"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=rot, yticklabels=rot,
           xlabel="Previsto", ylabel="Real", title=titulo)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    return ax


def plot_curvas(resultados: dict, y_true, titulo_sufixo=""):
    """resultados = {nome_modelo: score}. Desenha ROC e Precisão-Recall lado a lado."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for nome, s in resultados.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        a1.plot(fpr, tpr, label=f"{nome} (AUC={roc_auc_score(y_true, s):.3f})")
        p, r, _ = precision_recall_curve(y_true, s)
        a2.plot(r, p, label=f"{nome} (AP={average_precision_score(y_true, s):.3f})")
    a1.plot([0, 1], [0, 1], "k:", lw=1)
    a1.set(xlabel="Taxa de falso menor (FPR)", ylabel="Recall de menores (TPR)", title=f"Curva ROC {titulo_sufixo}")
    a2.axvline(RECALL_ALVO, ls="--", c="red", lw=1, label=f"recall = {RECALL_ALVO}")
    a2.set(xlabel="Recall de menores", ylabel="Precisão", title=f"Curva Precisão-Recall {titulo_sufixo}")
    a1.legend(fontsize=8); a2.legend(fontsize=8)
    return fig
