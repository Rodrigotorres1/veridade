"""Fábrica dos modelos clássicos + espaços de busca do Optuna.

Todos recebem os 512 embeddings FaRL e predizem P(menor de 18).
Ficar num lugar só garante que notebook, retreino no Airflow e API usem a mesma definição.
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
BASELINE = "Regressao Logistica"
MODELOS = ["KNN", "Arvore de Decisao", "Random Forest", "AdaBoost", "MLP"]

# Parâmetros iniciais (antes da otimização), escolhidos por bom senso e justificados no relatório
PADRAO = {
    BASELINE: {"C": 1.0},
    "KNN": {"n_neighbors": 25, "weights": "distance", "pca": None},
    "Arvore de Decisao": {"max_depth": 12, "min_samples_leaf": 10, "criterion": "gini"},
    "Random Forest": {"n_estimators": 300, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 2},
    "AdaBoost": {"n_estimators": 200, "learning_rate": 0.5, "profundidade_base": 1},
    "MLP": {"camadas": "256", "alpha": 1e-4, "learning_rate_init": 1e-3},
}


def criar_modelo(nome: str, **params) -> Pipeline:
    p = {**PADRAO[nome], **params}
    if nome == BASELINE:
        # class_weight compensa o desbalanceamento (só ~18% de menores)
        return Pipeline([("escala", StandardScaler()),
                         ("modelo", LogisticRegression(C=p["C"], class_weight="balanced", max_iter=2000))])
    if nome == "KNN":
        # embeddings já têm norma 1: distância euclidiana equivale à distância de cosseno
        passos = [("pca", PCA(n_components=p["pca"], random_state=SEED))] if p["pca"] else []
        return Pipeline(passos + [("modelo", KNeighborsClassifier(
            n_neighbors=p["n_neighbors"], weights=p["weights"], n_jobs=-1))])
    if nome == "Arvore de Decisao":
        return Pipeline([("modelo", DecisionTreeClassifier(
            max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], criterion=p["criterion"],
            class_weight="balanced", random_state=SEED))])
    if nome == "Random Forest":
        return Pipeline([("modelo", RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], max_features=p["max_features"],
            min_samples_leaf=p["min_samples_leaf"], class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED))])
    if nome == "AdaBoost":
        return Pipeline([("modelo", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=p["profundidade_base"]),
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], random_state=SEED))])
    if nome == "MLP":
        camadas = tuple(int(c) for c in str(p["camadas"]).split("-"))
        return Pipeline([("escala", StandardScaler()), ("modelo", MLPClassifier(
            hidden_layer_sizes=camadas, alpha=p["alpha"], learning_rate_init=p["learning_rate_init"],
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=300,
            random_state=SEED))])
    raise ValueError(f"Modelo desconhecido: {nome}")


def espaco_busca(trial, nome: str) -> dict:
    """Espaço de hiperparâmetros do Optuna para cada modelo."""
    if nome == BASELINE:
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if nome == "KNN":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 3, 75),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "pca": trial.suggest_categorical("pca", [None, 32, 64, 128, 256])}
    if nome == "Arvore de Decisao":
        return {"max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])}
    if nome == "Random Forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.1, 0.3]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)}
    if nome == "AdaBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
                "profundidade_base": trial.suggest_int("profundidade_base", 1, 3)}
    if nome == "MLP":
        return {"camadas": trial.suggest_categorical("camadas", ["256", "512-256", "256-128"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)}
    raise ValueError(nome)
