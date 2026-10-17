#! /home/dominique/spaceship-titanic/.venv/bin/python3
# entraîner les modèles.
# V1 : drop Surname, compare Logistic vs HGB vs CatBoost vs LightGBM vs XGBoost.
# Protocole honnête : split 3 voies (fit 60 / blend 20 / test 20 verrouillé),
# le blend gagnant est choisi sur blend puis mesuré une fois sur test.
# Sauve le blend ré-entraîné sur tout le train.

import importlib.util
from itertools import combinations
from pathlib import Path

import colorama
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from colorama import Fore, Style
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

colorama.init(autoreset=True, wrap=True)


def header(title: str, color: str = Fore.CYAN) -> None:
    bar = "=" * 16
    line = f"{bar} {title} {bar}"
    print(color + Style.BRIGHT + "\n" + line + Style.RESET_ALL)
    print(color + "-" * len(line) + Style.RESET_ALL)


def info(msg: str) -> None:
    print(Fore.WHITE + msg + Style.RESET_ALL)


def step(msg: str) -> None:
    print(Fore.YELLOW + Style.BRIGHT + msg + Style.RESET_ALL)


DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Étape 0 : charge 02_features.py (nom de fichier non importable directement car commence par un chiffre).
_spec = importlib.util.spec_from_file_location(
    "features02", Path(__file__).parent / "02_features.py"
)
features02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features02)

# Étape 1 : listes de colonnes V1 (Surname droppé : 8473 uniques, voir EDA).
CAT = ["HomePlanet", "Destination", "Deck", "Side"]
NUM = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "CabinNum",
    "GroupSize",
    "TotalSpent",
]
BOOL = ["CryoSleep", "VIP", "Solo", "NoSpending"]
TARGET = "Transported"
DROP = ["Surname"]


def build_pipelines() -> dict:
    """Cinq pipelines comparés : Logistic (avec scaling) vs 4 boosters (sans)."""
    # Étape 2 : prétraitement Logistic = OneHot + scaling des numériques.
    pre_log = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", StandardScaler(), NUM),
        ],
        remainder="passthrough",  # BOOL passent tels quels.
    )
    pipe_log = Pipeline([("pre", pre_log), ("clf", LogisticRegression(max_iter=1000))])
    # Étape 3 : prétraitement arbres = OneHot seul (pas besoin de scaler).
    # Chaque modèle a sa propre instance (évite tout partage d'état au fit).
    pipe_hgb = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                    remainder="passthrough",  # NUM + BOOL passent tels quels.
                ),
            ),
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ]
    )
    pipe_cat = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                    remainder="passthrough",
                ),
            ),
            ("clf", CatBoostClassifier(random_seed=42, verbose=False)),
        ]
    )
    pipe_lgb = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                    remainder="passthrough",
                ),
            ),
            ("clf", LGBMClassifier(random_state=42, verbose=-1)),
        ]
    )
    pipe_xgb = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                    remainder="passthrough",
                ),
            ),
            ("clf", XGBClassifier(random_state=42, eval_metric="logloss")),
        ]
    )
    return {
        "logistic": pipe_log,
        "hgb": pipe_hgb,
        "catboost": pipe_cat,
        "lightgbm": pipe_lgb,
        "xgboost": pipe_xgb,
    }


if __name__ == "__main__":
    # Étape 4 : double split du brut AVANT learn_stats (anti-fuite).
    # fit (60 %) : entraîne les modèles. blend (20 %) : choisit la combinaison.
    # test (20 %) : mesure FINALE unique, verrouillée, jamais utilisée pour choisir.
    header("ÉTAPE 4 - DATA SPLIT", Fore.BLUE)
    raw = pd.read_csv(DATA_DIR / "train.csv")
    fit_raw, tmp_raw = train_test_split(
        raw, test_size=0.4, stratify=raw[TARGET], random_state=42
    )
    blend_raw, test_raw = train_test_split(
        tmp_raw, test_size=0.5, stratify=tmp_raw[TARGET], random_state=42
    )
    info(f"split: fit={len(fit_raw)} blend={len(blend_raw)} test={len(test_raw)}")
    stats_fit = features02.learn_stats(fit_raw)
    fit = features02.make_features(fit_raw, stats_fit)
    bl = features02.make_features(blend_raw, stats_fit)
    te = features02.make_features(test_raw, stats_fit)

    X_fit, y_fit = fit.drop(columns=[TARGET, *DROP]), fit[TARGET].astype(bool)
    X_bl, y_bl = bl.drop(columns=[TARGET, *DROP]), bl[TARGET].astype(bool)
    X_te, y_te = te.drop(columns=[TARGET, *DROP]), te[TARGET].astype(bool)

    # Étape 5 : entraîne les cinq sur fit, compare sur blend (choix, pas de mesure finale).
    header("ÉTAPE 5 - INDIVIDUAL MODEL TRAINING", Fore.GREEN)
    pipes = build_pipelines()

    # print("pipes TOTO", pipes)

    scores_blend = {}
    for name, pipe in pipes.items():
        step(f"  → Training {name} ...")
        pipe.fit(X_fit, y_fit)
        pred = pipe.predict(X_bl)
        acc = accuracy_score(y_bl, pred)
        scores_blend[name] = acc
        print(Fore.GREEN + f"\n== {name} : accuracy blend = {acc:.4f} ==")
        print(classification_report(y_bl, pred))

        # Affichage des attributs pour le modèle logistique
        if name == "logistic":
            log_model = pipe.named_steps["clf"]
            print("\nLogistic Regression - Attributs appris :")
            print(f"model.coef_          : {log_model.coef_}")
            print(f"model.intercept_     : {log_model.intercept_}")
            print(f"model.classes_       : {log_model.classes_}")
            print(f"model.n_features_in_ : {log_model.n_features_in_}")

    for name, pipe in pipes.items():
        try:
            check_is_fitted(pipe)
            print(f"{name}: entraîné")
        except Exception:
            print(f"{name}: pas entraîné")

    # Étape 6 : teste les 26 combinaisons sur blend, verrouille la gagnante.
    header("ÉTAPE 6 - BLEND COMBINATION SEARCH", Fore.MAGENTA)
    # Les probas viennent d'un jeu non vu au fit (indépendant de l'entraînement),
    # mais ce même jeu sert à CHOISIR : le score blend du gagnant est donc optimiste.
    model_names = [
        "lightgbm",
        "catboost",
        "xgboost",
        "hgb",
        "logistic",
    ]

    # Probabilités de la classe True sur le jeu blend
    probas_bl = {name: pipes[name].predict_proba(X_bl)[:, 1] for name in model_names}

    # Tester toutes les combinaisons de 2 à 5 modèles
    results = []

    for n in range(2, len(model_names) + 1):
        for combo in combinations(model_names, n):
            # Moyenne des probabilités des modèles sélectionnés
            blend_proba = np.mean(
                [probas_bl[name] for name in combo],
                axis=0,
            )

            pred = blend_proba >= 0.5
            acc = accuracy_score(y_bl, pred)

            results.append((acc, combo))
            print(
                f"  {Fore.MAGENTA}{'+'.join(combo):45s} -> {acc:.4f}{Style.RESET_ALL}"
            )

    # Meilleure combinaison sur blend (choix verrouillé ici, pas de retour en arrière)
    results.sort(reverse=True, key=lambda x: x[0])
    best_acc_blend, best_combo = results[0]
    best_combo = list(best_combo)

    print(Fore.MAGENTA + Style.BRIGHT + "\nMeilleures combinaisons (blend) :")
    for acc, combo in results[:5]:
        print(f"  {'+'.join(combo):45s} -> {acc:.4f}")
    print(
        f"\nCombinaison verrouillée : {'+'.join(best_combo)} ({best_acc_blend:.4f} sur blend)"
    )

    # Étape 7 : mesure FINALE unique sur test verrouillé (seul chiffre honnête).
    header("ÉTAPE 7 - FINAL TEST MEASUREMENT (VERROUILLÉ)", Fore.RED)
    info("Individuels :")
    for name, pipe in pipes.items():
        acc_te = accuracy_score(y_te, pipe.predict(X_te))
        print(
            f"  {Fore.CYAN}{name:15s}{Style.RESET_ALL} -> {Fore.RED}{acc_te:.4f}{Style.RESET_ALL}"
        )
    probas_te = {name: pipes[name].predict_proba(X_te)[:, 1] for name in best_combo}
    blend_proba_te = np.mean([probas_te[name] for name in best_combo], axis=0)
    acc_blend_te = accuracy_score(y_te, blend_proba_te >= 0.5)
    print(
        f"  {Fore.RED}{'+'.join(best_combo):15s} -> {acc_blend_te:.4f}  (score honnête){Style.RESET_ALL}"
    )

    # Étape 8 : ré-entraîne les modèles du combo gagnant sur tout le train et sauve.
    header("ÉTAPE 8 - RE-TRAIN ON FULL DATA & SAVE", Fore.CYAN)
    stats_full = features02.learn_stats(raw)
    full = features02.make_features(raw, stats_full)
    X_full, y_full = full.drop(columns=[TARGET, *DROP]), full[TARGET].astype(bool)
    saved_pipes = {}
    for name in best_combo:
        step(f"  → Re-training {name} on full train ...")
        pipes[name].fit(X_full, y_full)
        saved_pipes[name] = pipes[name]
    out = MODELS_DIR / "model.pkl"
    joblib.dump(
        {
            "pipelines": saved_pipes,
            "combo": best_combo,
            "stats": stats_full,
            "score_blend": best_acc_blend,
            "score_test": acc_blend_te,
        },
        out,
    )
    info(f"Blend sauvé : {out} | combo : {best_combo}")
