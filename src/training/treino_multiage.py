"""Treino e predição do MultiAge.

Perda = entropia cruzada da idade (102 classes) + soma das BCE das cabeças binárias
(alpha = 1, gamma = 0 e age gap G = 0, a configuração padrão do artigo).
Reamostragem balanceada por idade (Eq. 7): cada uma das 12 faixas etárias tem a mesma
chance de aparecer no mini-batch.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.config import IDADE_MAX
from src.models.multiage import MultiAge

FAIXAS_REAMOSTRAGEM = [4, 8, 12, 16, 20, 24, 28, 32, 36, 42, 50]  # 12 faixas, como no artigo


def fixar_semente(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pesos_reamostragem(idades: np.ndarray) -> np.ndarray:
    faixa = np.digitize(idades, FAIXAS_REAMOSTRAGEM)
    contagem = np.bincount(faixa, minlength=len(FAIXAS_REAMOSTRAGEM) + 1)
    return 1.0 / contagem[faixa]


def _alvos_binarios(idades: torch.Tensor, limiares):
    return {str(t): (idades < t).float() for t in limiares}


def treinar_multiage(Xtr, idade_tr, Xva, idade_va, limiares, seed: int = 0, reamostrar: bool = True,
                     epocas: int = 60, paciencia: int = 8, lote: int = 256, lr: float = 1e-3,
                     weight_decay: float = 1e-4, device: str | None = None, verbose: bool = False):
    """Treina com early stopping pela perda de validação. Devolve (modelo, historico)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fixar_semente(seed)
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(idade_tr, dtype=torch.long).clamp(0, IDADE_MAX)
    Xva_t, yva_t = torch.tensor(Xva, dtype=torch.float32).to(device), torch.tensor(idade_va, dtype=torch.long).clamp(0, IDADE_MAX).to(device)

    gerador = torch.Generator().manual_seed(seed)
    if reamostrar:
        amostrador = WeightedRandomSampler(torch.tensor(pesos_reamostragem(idade_tr)), num_samples=len(Xtr),
                                           replacement=True, generator=gerador)
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=lote, sampler=amostrador)
    else:
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=lote, shuffle=True, generator=gerador)

    modelo = MultiAge(limiares).to(device)
    otim = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()

    def perda(xb, yb):
        lg_idade, lg_bin = modelo(xb)
        total = ce(lg_idade, yb)
        alvos = _alvos_binarios(yb, limiares)
        for t, lg in lg_bin.items():
            total = total + bce(lg, alvos[t])
        return total

    historico, melhor, melhor_estado, sem_melhora = [], np.inf, None, 0
    for ep in range(epocas):
        modelo.train(); soma, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            otim.zero_grad(); l = perda(xb, yb); l.backward(); otim.step()
            soma += l.item() * len(xb); n += len(xb)
        modelo.eval()
        with torch.no_grad():
            l_val = perda(Xva_t, yva_t).item()
        historico.append({"epoca": ep + 1, "perda_treino": soma / n, "perda_val": l_val})
        if verbose:
            print(f"época {ep + 1:2d} | treino {soma / n:.4f} | val {l_val:.4f}")
        if l_val < melhor - 1e-4:
            melhor, melhor_estado, sem_melhora = l_val, copy.deepcopy(modelo.state_dict()), 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                break
    modelo.load_state_dict(melhor_estado)
    return modelo.cpu().eval(), historico


def prever_scores(modelo: MultiAge, X: np.ndarray, limiares_avaliacao) -> dict:
    """Score de "menoridade" para cada limiar avaliado (maior = mais provável ser menor).

    Se o modelo tem cabeça dedicada ao limiar T, usa P(menor que T) dessa cabeça.
    Senão, usa a idade estimada (score = -idade), como o modelo "só idade" do artigo.
    """
    saida = modelo.prever(torch.tensor(X, dtype=torch.float32))
    idade = saida["idade"].numpy()
    scores = {"idade": idade}
    for t in limiares_avaliacao:
        chave = f"p_menor_{t}"
        scores[t] = saida[chave].numpy() if chave in saida else -idade
    return scores
