#! /home/dominique/spaceship-titanic/.venv/bin/python3
# entraîner les modèles.
# V3 : sélection du blend sur probabilités OOF GROUPÉES (StratifiedGroupKFold) + log-loss.
#
# Historique :
#   V1 : choix du blend sur un split "blend" de 1738 lignes à l'accuracy. Les écarts
#        entre combinaisons (0.8022 -> 0.8051) valaient ~0.3 SE : du bruit.
#   V2 : choix sur probas OOF (5-fold aléatoire) au log-loss, sur 6954 lignes.
#   V3 : les folds gardent les GROUPES (une famille = même GroupId) du même côté.
#
# Pourquoi grouper : une famille partage Cabin, GroupSize, Surname... Si deux membres
# d'un même groupe tombent l'un dans le train et l'autre dans la validation, le modèle
# peut partiellement mémoriser ce groupe -> probas OOF optimistes. Avec
# StratifiedGroupKFold, un groupe est entièrement d'un côté ou de l'autre :
#   - plus de fuite de groupe,
#   - et GroupSize/Solo (calculés dans le sous-ensemble) deviennent cohérents entre
#     train et validation, alors qu'un split aléatoire leur donnait des valeurs
#     différentes pour un même groupe (ex. 3 en train, 1 en validation).
#
# Protocole :
#   dev  (80 %) : StratifiedGroupKFold -> probas hors-échantillon (OOF) pour les 5 modèles.
#                 Les stats de features (learn_stats) sont ré-apprises dans chaque
#                 fold sur le train du fold uniquement (anti-fuite).
#   test (20 %) : jamais utilisé pour choisir. Mesure finale unique.
#   fin         : les modèles du combo gagnant sont ré-entraînés sur tout le train
#                 et sauvegardés avec leurs poids.

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
from scipy.optimize import minimize
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

colorama.init(autoreset=True, wrap=True)

N_SPLITS = 5  # folds de la cross-validation qui fabrique les probas OOF
SEED = 42  # graine unique : tout est reproductible
EPS = 1e-15  # clip des probas avant log-loss (log(0) interdit)


def header(title: str, color: str = Fore.CYAN) -> None:
    bar = "=" * 16
    line = f"{bar} {title} {bar}"
    print(color + Style.BRIGHT + "\n" + line + Style.RESET_ALL)
    print(color + "-" * len(line) + Style.RESET_ALL)


def info(msg: str) -> None:
    print(Fore.WHITE + msg + Style.RESET_ALL)


def step(msg: str) -> None:
    print(Fore.YELLOW + Style.BRIGHT + msg + Style.RESET_ALL)


def show(nom: str, m: dict, color: str = Fore.WHITE) -> None:
    """Une ligne du tableau de résultats : log-loss, AUC et accuracy (seuil 0.5)."""
    print(
        f"  {color}{nom:38s}{Style.RESET_ALL}"
        f" logloss={m['logloss']:.5f}  auc={m['auc']:.4f}  acc={m['acc']:.4f}"
    )


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
FEATURES_DROP = [TARGET, *DROP]


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
            ("clf", HistGradientBoostingClassifier(random_state=SEED)),
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
            ("clf", CatBoostClassifier(random_seed=SEED, verbose=False)),
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
            ("clf", LGBMClassifier(random_state=SEED, verbose=-1)),
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
            ("clf", XGBClassifier(random_state=SEED, eval_metric="logloss")),
        ]
    )
    return {
        "logistic": pipe_log,
        "hgb": pipe_hgb,
        "catboost": pipe_cat,
        "lightgbm": pipe_lgb,
        "xgboost": pipe_xgb,
    }


def group_ids(df: pd.DataFrame) -> np.ndarray:
    """GroupId = partie avant le '_' du PassengerId (une famille voyage ensemble).

    Même définition que dans 02_features.py : c'est la clé qui doit rester entière
    du même côté d'un fold, sinon l'OOF est optimiste.
    """
    return df["PassengerId"].str.split("_").str[0].to_numpy()


def metrics(y_true, proba) -> dict:
    """Log-loss, AUC et accuracy (seuil 0.5) d'un vecteur de probabilités."""
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(proba, dtype=float), EPS, 1 - EPS)
    return {
        "logloss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "acc": float(accuracy_score(y, p >= 0.5)),
    }


def uniform(names) -> dict:
    """Poids de la moyenne simple (ce que faisait la V1)."""
    return {name: 1.0 / len(names) for name in names}


def combine(probas: dict, names, weights: dict) -> np.ndarray:
    """Combinaison linéaire des probas : somme(w_i * p_i)."""
    return np.sum([weights[name] * probas[name] for name in names], axis=0)


def optimise_weights(probas: dict, y_true, combo) -> dict:
    """Cherche les poids (>=0, somme=1) qui minimisent le log-loss.

    Le log-loss est l'objectif naturel d'une moyenne de probas : c'est une
    fonction lisse, donc beaucoup plus stable que l'accuracy pour choisir.
    """
    P = np.column_stack([probas[name] for name in combo])
    y = np.asarray(y_true).astype(int)
    k = len(combo)

    def loss(w: np.ndarray) -> float:
        w = np.clip(w, 0, None)
        total = w.sum()
        w = w / total if total > 0 else np.full(k, 1.0 / k)
        return log_loss(y, np.clip(P @ w, EPS, 1 - EPS))

    # Départ = moyenne uniforme : on ne peut pas faire pire que ce point.
    res = minimize(
        loss,
        np.full(k, 1.0 / k),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-9},
    )
    w = np.clip(res.x, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)
    return {name: float(wi) for name, wi in zip(combo, w)}


def mcnemar(y_true, pred_a, pred_b) -> tuple:
    """Test exact de McNemar entre deux prédicteurs sur le MÊME jeu.

    Répond à : "l'écart observé est-il significatif, ou du bruit d'échantillon ?"
    """
    y = np.asarray(y_true).astype(bool)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    a_seul = int(np.sum(a_ok & ~b_ok))  # A a raison, B a tort
    b_seul = int(np.sum(~a_ok & b_ok))  # A a tort, B a raison
    if a_seul + b_seul == 0:
        return 1.0, a_seul, b_seul
    p = binomtest(min(a_seul, b_seul), a_seul + b_seul, 0.5).pvalue
    return float(p), a_seul, b_seul


if __name__ == "__main__":
    # Étape 4 : split dev (80 %) / test (20 % verrouillé).
    # Même graine et même double split que la V1 : le jeu test est IDENTIQUE,
    # donc les chiffres restent comparables d'une version à l'autre.
    header("ÉTAPE 4 - SPLIT (dev 80 % / test 20 % verrouillé)", Fore.BLUE)
    raw = pd.read_csv(DATA_DIR / "train.csv")
    fit_raw, tmp_raw = train_test_split(
        raw, test_size=0.4, stratify=raw[TARGET], random_state=SEED
    )
    blend_raw, test_raw = train_test_split(
        tmp_raw, test_size=0.5, stratify=tmp_raw[TARGET], random_state=SEED
    )
    dev_raw = pd.concat([fit_raw, blend_raw], ignore_index=True)
    y_dev = dev_raw[TARGET].astype(bool).to_numpy()
    y_te = test_raw[TARGET].astype(bool).to_numpy()
    info(f"dev={len(dev_raw)} (probas OOF sur {N_SPLITS} folds) | test={len(test_raw)} verrouillé")

    # Diagnostic : ce split dev/test reste volontairement aléatoire (jeu test identique
    # à la V1/V2 -> chiffres comparables). Il reste donc une fuite résiduelle possible
    # entre dev et test : passagers de test dont le groupe (ou la cabin) est déjà dans dev.
    g_dev = group_ids(dev_raw)
    g_te = group_ids(test_raw)
    grp_partages = int(np.isin(g_te, np.unique(g_dev)).sum())
    cab_dev = set(dev_raw["Cabin"].dropna())
    cab_partagees = int(test_raw["Cabin"].isin(cab_dev).sum())
    info(
        f"fuite dev->test : {grp_partages}/{len(test_raw)} passagers de test partagent un "
        f"groupe avec dev, {cab_partagees} partagent une cabin"
    )

    # Étape 5 : fabrique les probas OOF. Chaque ligne de dev est prédite par un
    # modèle qui ne l'a jamais vue, et dont le fold de validation ne contient
    # aucun groupe présent dans son train.
    header(f"ÉTAPE 5 - PROBAS OOF ({N_SPLITS}-fold GROUPÉ sur dev)", Fore.GREEN)
    names = list(build_pipelines().keys())
    oof = {name: np.zeros(len(dev_raw)) for name in names}
    groups_dev = group_ids(dev_raw)
    uniq, counts = np.unique(groups_dev, return_counts=True)
    info(
        f"dev : {len(dev_raw)} passagers en {len(uniq)} groupes "
        f"(taille médiane={int(np.median(counts))}, max={int(counts.max())})"
    )
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for k, (tr_idx, va_idx) in enumerate(
        sgkf.split(dev_raw, y_dev, groups=groups_dev), start=1
    ):
        # Contrôle : aucun groupe ne doit être à cheval sur train et validation.
        partages = set(groups_dev[tr_idx]) & set(groups_dev[va_idx])
        assert not partages, f"groupe partagé train/val : {sorted(partages)[:3]}"
        step(
            f"  → fold {k}/{N_SPLITS} : fit={len(tr_idx)} val={len(va_idx)} "
            f"({len(np.unique(groups_dev[va_idx]))} groupes en validation)"
        )
        # stats apprises sur le train du fold seulement (anti-fuite).
        stats = features02.learn_stats(dev_raw.iloc[tr_idx])
        X_tr = features02.make_features(dev_raw.iloc[tr_idx], stats).drop(columns=FEATURES_DROP)
        X_va = features02.make_features(dev_raw.iloc[va_idx], stats).drop(columns=FEATURES_DROP)
        y_tr = y_dev[tr_idx]
        for name, pipe in build_pipelines().items():
            pipe.fit(X_tr, y_tr)
            oof[name][va_idx] = pipe.predict_proba(X_va)[:, 1]
    info(f"OOF complet : {int(sum((oof[n] > 0).sum() for n in names) / len(names))}/{len(dev_raw)} lignes prédites")
    info(f"contrôle groupes : 0 groupe partagé train/val sur les {N_SPLITS} folds (assert)")

    # Étape 6 : niveau de référence de chaque modèle seul, sur les probas OOF.
    header("ÉTAPE 6 - INDIVIDUELS (OOF)", Fore.GREEN)
    m_oof = {name: metrics(y_dev, oof[name]) for name in names}
    for name in sorted(names, key=lambda n: m_oof[n]["logloss"]):
        show(name, m_oof[name], Fore.CYAN)
    best_single = min(names, key=lambda n: m_oof[n]["logloss"])
    info(f"meilleur individuel (log-loss OOF) : {best_single}")

    # Étape 7 : teste les 26 combinaisons (2 à 5 modèles) SUR LES PROBAS OOF,
    # classées au log-loss. 6954 lignes au lieu de 1738 : 4x plus de signal.
    header("ÉTAPE 7 - RECHERCHE DU BLEND (26 combinaisons, OOF, log-loss)", Fore.MAGENTA)
    results = []
    for n in range(2, len(names) + 1):
        for combo in combinations(names, n):
            acc_ll = metrics(y_dev, combine(oof, list(combo), uniform(combo)))
            results.append((acc_ll["logloss"], combo, acc_ll))
    results.sort(key=lambda x: x[0])

    print(Fore.MAGENTA + Style.BRIGHT + "Top 5 au log-loss (moyenne uniforme) :")
    for ll, combo, m in results[:5]:
        show("+".join(combo), m, Fore.MAGENTA)

    # Comparaison pédagogique : le classement par accuracy est bien plus bruité.
    par_acc = sorted(results, key=lambda x: -x[2]["acc"])
    print(Fore.MAGENTA + Style.BRIGHT + "\nTop 5 à l'accuracy (pour comparer) :")
    for ll, combo, m in par_acc[:5]:
        show("+".join(combo), m, Fore.MAGENTA)

    best_ll, best_combo, best_m = results[0]
    best_combo = list(best_combo)
    print(
        Fore.MAGENTA
        + Style.BRIGHT
        + f"\nCombinaison verrouillée : {'+'.join(best_combo)} (logloss OOF={best_ll:.5f}, acc OOF={best_m['acc']:.4f})"
    )

    # Étape 8 : les poids. Optimisés sur les probas OOF au log-loss ; on garde la
    # moyenne uniforme si l'optimisation n'apporte rien (évite de suroptimiser).
    header("ÉTAPE 8 - POIDS DU BLEND (OOF)", Fore.MAGENTA)
    w_uni = uniform(best_combo)
    w_opt = optimise_weights(oof, y_dev, best_combo)
    m_opt = metrics(y_dev, combine(oof, best_combo, w_opt))
    show("moyenne uniforme", best_m, Fore.WHITE)
    show("poids optimisés", m_opt, Fore.MAGENTA)
    if m_opt["logloss"] < best_m["logloss"]:
        weights, m_ref, choix = w_opt, m_opt, "poids optimisés"
    else:
        weights, m_ref, choix = w_uni, best_m, "moyenne uniforme"
    info("poids retenus : " + ", ".join(f"{n}={weights[n]:.3f}" for n in best_combo))
    info(f"→ verrouillé : {'+'.join(best_combo)} ({choix}) logloss OOF={m_ref['logloss']:.5f}")

    # Étape 9 : mesure FINALE unique sur test verrouillé (seul chiffre honnête).
    # Les modèles sont ré-entraînés sur tout dev (80 %) avant d'être mesurés.
    header("ÉTAPE 9 - MESURE FINALE SUR TEST VERROUILLÉ", Fore.RED)
    stats_dev = features02.learn_stats(dev_raw)
    dev = features02.make_features(dev_raw, stats_dev)
    X_dev, y_dev_full = dev.drop(columns=FEATURES_DROP), dev[TARGET].astype(bool)
    X_te = features02.make_features(test_raw, stats_dev).drop(columns=FEATURES_DROP)

    pipes_dev = {}
    for name, pipe in build_pipelines().items():
        step(f"  → fit {name} sur dev ({len(dev_raw)}) ...")
        pipe.fit(X_dev, y_dev_full)
        pipes_dev[name] = pipe
        if name == "logistic":
            # Attributs appris (pédagogique).
            log_model = pipe.named_steps["clf"]
            info(f"logistic coef_ (5 premiers) : {np.round(log_model.coef_[0][:5], 4)}")
            info(f"logistic intercept_         : {np.round(log_model.intercept_, 4)}")
            info(f"logistic classes_           : {log_model.classes_}")

    probas_te = {name: pipes_dev[name].predict_proba(X_te)[:, 1] for name in names}
    print(Fore.WHITE + Style.BRIGHT + "Individuels (test) :")
    m_te = {}
    for name in sorted(names, key=lambda n: m_oof[n]["logloss"]):
        m_te[name] = metrics(y_te, probas_te[name])
        show(name, m_te[name], Fore.CYAN)

    p_blend_te = combine(probas_te, best_combo, weights)
    m_blend_te = metrics(y_te, p_blend_te)
    show("+".join(best_combo), m_blend_te, Fore.RED)
    ci = 1.96 * np.sqrt(m_blend_te["acc"] * (1 - m_blend_te["acc"]) / len(test_raw))
    info(f"n={len(test_raw)} → accuracy ± {ci * 100:.2f} pts (IC 95 %) : tout écart plus petit est du bruit")

    # Test de McNemar : blend vs meilleur individuel, sur le test verrouillé.
    p_val, a_seul, b_seul = mcnemar(
        y_te, p_blend_te >= 0.5, probas_te[best_single] >= 0.5
    )
    info(
        f"McNemar blend vs {best_single} : p={p_val:.3f} "
        f"(blend seul correct={a_seul}, {best_single} seul correct={b_seul})"
    )
    if p_val > 0.05:
        info("→ écart NON significatif : les deux se valent sur ce jeu de test.")
    else:
        info("→ écart significatif au seuil 5 %.")

    # Étape 10 : sauve les probas (OOF + test) pour que 04_evaluate.py puisse faire
    # l'analyse et les graphiques SANS ré-entraîner quoi que ce soit.
    header("ÉTAPE 10 - SAUVEGARDE DES PROBAS (pour 04_evaluate.py)", Fore.CYAN)
    eval_out = MODELS_DIR / "eval.pkl"
    joblib.dump(
        {
            "protocol": f"OOF {N_SPLITS}-fold StratifiedGroupKFold / selection log-loss / seed {SEED}",
            "models": names,
            "combo": best_combo,
            "weights": weights,
            "oof": oof,  # {modèle: probas OOF sur dev}
            "y_dev": y_dev,
            "group_dev": groups_dev,  # GroupId de chaque ligne de dev
            "passenger_dev": dev_raw["PassengerId"].to_numpy(),
            "test": probas_te,  # {modèle: probas sur le test verrouillé}
            "y_test": y_te,
            "group_test": g_te,
            "passenger_test": test_raw["PassengerId"].to_numpy(),
            "metrics_oof": m_ref,
            "metrics_test": m_blend_te,
        },
        eval_out,
    )
    info(f"Probas sauvées : {eval_out}")

    # Étape 11 : ré-entraîne les modèles du combo sur TOUT le train et sauve.
    header("ÉTAPE 11 - RE-TRAIN SUR TOUT LE TRAIN & SAVE", Fore.CYAN)
    stats_full = features02.learn_stats(raw)
    full = features02.make_features(raw, stats_full)
    X_full, y_full = full.drop(columns=FEATURES_DROP), full[TARGET].astype(bool)
    saved_pipes = {}
    for name in best_combo:
        step(f"  → Re-training {name} on full train ...")
        pipe = build_pipelines()[name]
        pipe.fit(X_full, y_full)
        saved_pipes[name] = pipe
    out = MODELS_DIR / "model.pkl"
    joblib.dump(
        {
            "pipelines": saved_pipes,
            "combo": best_combo,
            "weights": weights,
            "stats": stats_full,
            "metrics_oof": m_ref,
            "metrics_test": m_blend_te,
            "score_blend": m_ref["acc"],  # compat V1 : score de sélection (ici OOF)
            "score_test": m_blend_te["acc"],  # compat V1 : score honnête
            "protocol": f"OOF {N_SPLITS}-fold StratifiedGroupKFold / selection log-loss / seed {SEED}",
        },
        out,
    )
    info(f"Blend sauvé : {out} | combo : {best_combo}")
    info("poids : " + ", ".join(f"{n}={weights[n]:.3f}" for n in best_combo))
