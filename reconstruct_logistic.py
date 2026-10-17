#! /home/dominique/spaceship-titanic/.venv/bin/python3
# Reconstitution explicite de la fonction de décision de la LogisticRegression
# telle qu'elle est construite dans 03_train.py (mêmes colonnes, même prétraitement).
#
#   z = w . phi(x) + b
#   p(Transported=1) = sigmoid(z) = 1 / (1 + exp(-z))
#
# où phi(x) = sortie du ColumnTransformer :
#   [ one-hot(CAT) | StandardScaler(NUM) | BOOL (0/1) ]
#
# Le script replie aussi le StandardScaler dans les coefficients pour donner
# la formule DIRECTEMENT sur les variables brutes (x tel que lu dans le CSV).

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent

# Étape 0 : charge 02_features.py (nom non importable, commence par un chiffre).
_spec = importlib.util.spec_from_file_location("features02", HERE / "02_features.py")
features02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features02)

# Étape 1 : mêmes listes de colonnes que 03_train.py.
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
FEATURES_DROP = [TARGET, "Surname"]


def build_logistic() -> Pipeline:
    """Pipeline identique à 03_train.py (bloc 'logistic')."""
    pre_log = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", StandardScaler(), NUM),
        ],
        remainder="passthrough",  # BOOL passent tels quels (True/False -> 1/0).
    )
    return Pipeline([("pre", pre_log), ("clf", LogisticRegression(max_iter=1000))])


def main() -> None:
    raw = pd.read_csv(HERE / "data" / "train.csv")
    stats = features02.learn_stats(raw)
    df = features02.make_features(raw, stats)
    X = df.drop(columns=FEATURES_DROP)
    y = df[TARGET].astype(bool).to_numpy()

    pipe = build_logistic()
    pipe.fit(X, y)

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    w = clf.coef_[0]
    b = float(clf.intercept_[0])

    names = pre.get_feature_names_out()

    print("=" * 78)
    print("FONCTION APPRISE (espace transformé phi(x))")
    print("=" * 78)
    print(f"n features transformées : {len(names)} | classes_ = {clf.classes_}")
    print(f"intercept_ (biais b)    : {b:+.6f}")
    print(f"sigmoid(z) prédit P(Transported=True)")

    print("\n--- coefficients triés par |w| ---")
    order = np.argsort(-np.abs(w))
    for i in order:
        print(f"  {names[i]:32s} w = {w[i]:+.5f}")

    # Étape 2 : replie le StandardScaler dans les coefficients.
    # phi_j = (x_j - mean_j) / scale_j
    #   =>  w_j * phi_j = (w_j/scale_j) * x_j  -  w_j*mean_j/scale_j
    scaler = pre.named_transformers_["num"]
    mean_, scale_ = scaler.mean_, scaler.scale_

    raw_coef = {}          # nom_brut -> coefficient sur x brut
    b_raw = b
    for i, name in enumerate(names):
        if name.startswith("num__"):
            j = NUM.index(name[len("num__"):])
            a = w[i] / scale_[j]
            raw_coef[NUM[j]] = a
            b_raw += -w[i] * mean_[j] / scale_[j]
        else:
            # cat__<Feat>_<valeur> ou remainder__<Bool>
            pretty = name.split("__", 1)[1]
            raw_coef[pretty] = w[i]

    print("\n" + "=" * 78)
    print("FONCTION SUR VARIABLES BRUTES (StandardScaler replié)")
    print("=" * 78)
    print(f"z = {b_raw:+.6f}")
    for name, a in sorted(raw_coef.items(), key=lambda kv: -abs(kv[1])):
        print(f"      {a:+.6f} * {name}")

    # Étape 3 : VÉRIFICATION numérique — la formule maison == predict_proba.
    Xt = pre.transform(X)
    z_manu = Xt @ w + b
    p_manu = 1.0 / (1.0 + np.exp(-z_manu))
    p_sk = pipe.predict_proba(X)[:, 1]
    print("\n" + "=" * 78)
    print("VÉRIFICATION (maison vs sklearn)")
    print("=" * 78)
    print(f"  max |p_manu - p_sklearn| = {np.max(np.abs(p_manu - p_sk)):.2e}")

    # Étape 4 : formule z en espace brut, recalculée depuis les poids repliés.
    z_raw = np.full(len(X), b_raw)
    for name, a in raw_coef.items():
        if name in BOOL:
            col = X[name].astype(float)                     # True/False -> 1/0
        elif name in NUM:
            col = X[name].astype(float)
        else:  # "Feat_valeur" -> indicateur de la catégorie
            feat, val = name.split("_", 1)
            col = (X[feat].astype(str) == val).astype(float)
        z_raw += a * col
    p_raw = 1.0 / (1.0 + np.exp(-z_raw))
    print(f"  max |p_brut - p_sklearn| = {np.max(np.abs(p_raw - p_sk)):.2e}")

    # Étape 5 : un exemple concret (première ligne).
    print("\n" + "=" * 78)
    print("EXEMPLE (première ligne du train)")
    print("=" * 78)
    print(f"  z            = {z_manu[0]:+.6f}")
    print(f"  p(Transported)= {p_manu[0]:.4f}  (sklearn {p_sk[0]:.4f})")
    print(f"  y réel        = {bool(y[0])}")


if __name__ == "__main__":
    main()
