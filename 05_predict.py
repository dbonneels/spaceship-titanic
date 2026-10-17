#! /home/dominique/spaceship-titanic/.venv/bin/python3
# produire le fichier de soumission.
#
# Lit models/model.pkl (écrit par 03_train.py) : AUCUN ré-entraînement.
# Le pkl contient les modèles du blend gagnant ré-entraînés sur TOUT le train, plus :
#   - combo   : les modèles du blend verrouillé (ex. hgb + catboost)
#   - weights : leurs poids (ici 0.19 / 0.81, pas une moyenne uniforme)
#   - stats   : les imputations apprises sur le train (à réutiliser TELLES QUELLES)
#
# Garde-fou principal : les pipelines utilisent ColumnTransformer(remainder="passthrough"),
# qui recopie les colonnes restantes dans leur ordre d'origine. Si l'ordre des colonnes
# de X_test diffère de celui vu à l'entraînement, les prédictions sont fausses SANS
# lever d'erreur. D'où le contrôle explicite contre feature_names_in_.
#
# Sortie : submission.csv (PassengerId,Transported) au format de data/sample_submission.csv

import importlib.util
from pathlib import Path

import colorama
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # WSL sans écran : on sauve en PNG, plt.show() ne marche pas.
import matplotlib.pyplot as plt
from colorama import Fore, Style

colorama.init(autoreset=True, wrap=True)

# Seuil de décision. Le seuil "optimal" trouvé sur l'OOF valait 0.51, ce qui ne
# rapportait que +0.23 pt sur le test verrouillé, soit l'épaisseur du bruit
# (voir 04_evaluate.py, graphique 04_threshold.png) : on garde 0.50.
SEUIL = 0.50
N_ATTENDU = 4277  # lignes de test.csv / sample_submission.csv
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
PLOTS_DIR = BASE / "plots"
OUT = BASE / "submission.csv"
DROP = ["Surname", "Transported"]  # retirées avant d'entrer dans le modèle


def header(title: str, color: str = Fore.CYAN) -> None:
    bar = "=" * 16
    line = f"{bar} {title} {bar}"
    print(color + Style.BRIGHT + "\n" + line + Style.RESET_ALL)
    print(color + "-" * len(line) + Style.RESET_ALL)


def info(msg: str) -> None:
    print(Fore.WHITE + msg + Style.RESET_ALL)


def step(msg: str) -> None:
    print(Fore.YELLOW + Style.BRIGHT + msg + Style.RESET_ALL)


def warn(msg: str) -> None:
    print(Fore.RED + Style.BRIGHT + msg + Style.RESET_ALL)


# Charge 02_features.py (nom de fichier non importable directement car commence par un chiffre).
_spec = importlib.util.spec_from_file_location("features02", BASE / "02_features.py")
features02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features02)


def main() -> None:
    # Étape 1 : le modèle et son contrat.
    header("ÉTAPE 1 - CHARGEMENT DU MODÈLE", Fore.BLUE)
    model_path = MODELS_DIR / "model.pkl"
    if not model_path.exists():
        raise SystemExit(f"{model_path} absent -> lance d'abord 03_train.py")
    d = joblib.load(model_path)
    combo, weights = d["combo"], d["weights"]
    info(f"protocole : {d['protocol']}")
    info(f"combo     : {'+'.join(combo)}   poids " + ", ".join(f"{n}={weights[n]:.3f}" for n in combo))
    info(f"scores    : OOF acc={d['score_blend']:.4f} | test verrouillé acc={d['score_test']:.4f}")
    info(f"stats     : {d['stats']}")

    # Étape 2 : features du test, avec les stats apprises sur le train.
    header("ÉTAPE 2 - FEATURES DU TEST", Fore.GREEN)
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    info(f"test.csv  : {test_raw.shape[0]} lignes x {test_raw.shape[1]} colonnes")
    feats = features02.make_features(test_raw, d["stats"])
    X_te = feats.drop(columns=[c for c in DROP if c in feats.columns])
    info(f"X_test    : {X_te.shape[0]} lignes x {X_te.shape[1]} colonnes")

    # Garde-fou : même liste ET même ordre de colonnes que celles vues à l'entraînement.
    for name in combo:
        attendu = list(getattr(d["pipelines"][name], "feature_names_in_", X_te.columns))
        if list(X_te.columns) != attendu:
            warn(f"  {name} attend : {attendu}")
            warn(f"  X_test fournit : {list(X_te.columns)}")
            raise SystemExit("Contrat de colonnes rompu -> écriture annulée")
    info(f"contrat de colonnes : OK, {len(X_te.columns)} colonnes dans le bon ordre")
    if int(X_te.isna().sum().sum()) != 0:
        raise SystemExit("NaN dans X_test -> écriture annulée")

    # Étape 3 : probas de chaque modèle du combo, puis blend pondéré.
    header("ÉTAPE 3 - PRÉDICTION (blend pondéré)", Fore.MAGENTA)
    probas = {}
    for name in combo:
        probas[name] = d["pipelines"][name].predict_proba(X_te)[:, 1]
        step(f"  → {name:9s} : proba moyenne = {probas[name].mean():.4f}")
    p = np.sum([weights[name] * probas[name] for name in combo], axis=0)
    pred = (p >= SEUIL).astype(bool)
    info(f"seuil de décision      : {SEUIL:.2f}")
    info(f"proba moyenne du blend : {p.mean():.4f}")
    info(f"prédictions True       : {int(pred.sum())} / {len(pred)} ({100 * pred.mean():.2f} %)")
    for name in combo:
        accord = float(np.mean((probas[name] >= SEUIL) == pred))
        info(f"  accord blend / {name:9s} : {100 * accord:.2f} %")

    # Étape 4 : contrôles de forme avant d'écrire quoi que ce soit.
    header("ÉTAPE 4 - CONTRÔLES AVANT ÉCRITURE", Fore.YELLOW)
    sub = pd.DataFrame({"PassengerId": test_raw["PassengerId"], "Transported": pred})
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    problemes = []
    if len(sub) != N_ATTENDU or len(sample) != N_ATTENDU:
        problemes.append(f"lignes : soumission={len(sub)}, sample={len(sample)} (attendu {N_ATTENDU})")
    if list(sub["PassengerId"]) != list(sample["PassengerId"]):
        problemes.append("les PassengerId ne sont pas identiques à sample_submission.csv")
    if sub["PassengerId"].duplicated().any():
        problemes.append("PassengerId dupliqués")
    if sub["Transported"].isna().any():
        problemes.append("Transported contient des NaN")
    if problemes:
        for p_ in problemes:
            warn(f"  ✗ {p_}")
        raise SystemExit("contrôles échoués -> écriture annulée")
    info(f"  ✓ {len(sub)} lignes, 0 NaN, ids identiques à sample_submission.csv")
    info(f"  ✓ dtype Transported = {sub['Transported'].dtype}")

    # Étape 5 : écriture + relecture.
    header("ÉTAPE 5 - ÉCRITURE DE submission.csv", Fore.CYAN)
    sub.to_csv(OUT, index=False)
    relu = pd.read_csv(OUT)
    info(f"écrit  : {OUT}")
    info(f"relu   : {relu.shape[0]} lignes x {relu.shape[1]} colonnes | {list(relu.columns)}")
    info(relu.head(5).to_string(index=False))
    if len(relu) != len(sub) or list(relu.columns) != ["PassengerId", "Transported"]:
        raise SystemExit("la relecture ne correspond pas à ce qui a été écrit")

    # Distribution des probas : montre à quel point le modèle est tranché.
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.hist(p, bins=50, color="#2f7ed8")
    ax.axvline(SEUIL, color="#d64545", ls="--", lw=1.4, label=f"seuil = {SEUIL:.2f}")
    ax.set_title(f"Probas du blend sur test.csv - {'+'.join(combo)}")
    ax.set_xlabel("proba(Transported=True)")
    ax.set_ylabel("nombre de passagers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = PLOTS_DIR / "05_proba_hist.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    info(f"graphique : {plot_path}")

    header("RÉSUMÉ", Fore.CYAN)
    info(f"  submission.csv : {len(sub)} lignes, {100 * pred.mean():.2f} % de True")
    info(f"  blend {'+'.join(combo)} (poids " + ", ".join(f"{n}={weights[n]:.3f}" for n in combo) + f"), seuil {SEUIL:.2f}")
    warn("  rappel : le test verrouillé interne partageait ses familles avec dev (41 %),")
    warn("  alors que le vrai test Kaggle n'en partage aucune : le score réel sera plus bas")
    warn("  que 0.81 (plutôt ~0.79, voir 04_evaluate.py, étape 6).")


if __name__ == "__main__":
    main()
