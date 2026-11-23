"""Leitura dos rótulos a partir do nome dos arquivos de cada dataset.

UTKFace : "[idade]_[genero]_[raca]_[datahora].jpg.chip.jpg"  ex: 26_0_1_20170116.jpg.chip.jpg
FG-NET  : "[sujeito]A[idade][letra opcional].JPG"             ex: 001A02.JPG, 078A11a.JPG
"""
import re
from pathlib import Path

import pandas as pd

_UTK_RE = re.compile(r"^(\d+)_(\d)_(\d)_")
_FGNET_RE = re.compile(r"^(\d{3})A(\d{2})[a-z]?\.jpe?g$", re.IGNORECASE)
_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _imagens(pasta: Path):
    return sorted(p for p in Path(pasta).rglob("*") if p.suffix.lower() in _IMG_EXT)


def parse_utkface(pasta) -> pd.DataFrame:
    linhas, ignorados = [], 0
    for p in _imagens(pasta):
        m = _UTK_RE.match(p.name)
        if not m:  # alguns arquivos do UTKFace vêm sem gênero/raça
            ignorados += 1
            continue
        idade, genero, raca = map(int, m.groups())
        linhas.append({"arquivo": str(p), "dataset": "UTKFace", "idade": idade,
                       "genero": genero, "raca": raca, "sujeito": None})
    print(f"UTKFace: {len(linhas)} imagens lidas, {ignorados} ignoradas (nome fora do padrão)")
    return pd.DataFrame(linhas)


def parse_fgnet(pasta) -> pd.DataFrame:
    linhas, ignorados = [], 0
    for p in _imagens(pasta):
        m = _FGNET_RE.match(p.name)
        if not m:
            ignorados += 1
            continue
        sujeito, idade = m.groups()
        linhas.append({"arquivo": str(p), "dataset": "FG-NET", "idade": int(idade),
                       "genero": None, "raca": None, "sujeito": sujeito})
    print(f"FG-NET: {len(linhas)} imagens lidas, {ignorados} ignoradas")
    return pd.DataFrame(linhas)
