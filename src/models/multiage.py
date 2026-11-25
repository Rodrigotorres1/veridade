"""MultiAge: rede leve sobre os embeddings FaRL congelados (Gaul et al., 2025).

Arquitetura "Independent MLP" I-256 do artigo (Eq. 2 a 4):
    z (512) -> ReLU(W1 z) (256) -> ReLU(W2 h) (512) = o
    o -> cabeça de idade: softmax em 102 classes (0 a 101); idade = soma(a * p_a)
    o -> uma cabeça binária sigmoide por limiar T: P(idade < T)
"""
from __future__ import annotations

import torch
from torch import nn

from src.config import EMB_DIM, IDADE_MAX

N_CLASSES = IDADE_MAX + 1  # 102


class MultiAge(nn.Module):
    def __init__(self, limiares: list[int] | None = None, largura: int = 256, dropout: float = 0.1):
        super().__init__()
        self.limiares = list(limiares or [])
        self.tronco = nn.Sequential(
            nn.Linear(EMB_DIM, largura), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(largura, EMB_DIM), nn.ReLU(),
        )
        self.cabeca_idade = nn.Linear(EMB_DIM, N_CLASSES)
        self.cabecas_binarias = nn.ModuleDict({str(t): nn.Linear(EMB_DIM, 1) for t in self.limiares})
        self.register_buffer("idades", torch.arange(N_CLASSES, dtype=torch.float32))

    def forward(self, z):
        o = self.tronco(z)
        logits_idade = self.cabeca_idade(o)
        logits_bin = {t: cab(o).squeeze(-1) for t, cab in self.cabecas_binarias.items()}
        return logits_idade, logits_bin

    @torch.no_grad()
    def prever(self, z) -> dict:
        """Saída "de produção": idade estimada e P(menor que T) de cada cabeça."""
        self.eval()
        logits_idade, logits_bin = self(z)
        p = torch.softmax(logits_idade, dim=-1)
        saida = {"idade": (p * self.idades).sum(-1)}
        for t, lg in logits_bin.items():
            saida[f"p_menor_{t}"] = torch.sigmoid(lg)
        return saida
