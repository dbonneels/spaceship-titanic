#! /home/dominique/spaceship-titanic/.venv/bin/python3
# préparer les colonnes, notamment Cabin, PassengerId et les catégories.
# Option A : parsing + nettoyage seulement (encodage dans 03_train.py).

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # WSL sans écran : on sauve en PNG, plt.show() ne marche pas.
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def learn_stats(train: pd.DataFrame) -> dict:
    """Stats apprises sur train uniquement (anti-fuite)."""
    # Étape 0 : mémorise les valeurs de remplacement à partir du train seul.
    # On ne calcule jamais ces stats sur le test pour éviter la fuite de données.
    return {
        "age_median": train["Age"].median(),
        "home_mode": train["HomePlanet"].mode()[0],
        "dest_mode": train["Destination"].mode()[0],
    }


def make_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Même règles pour train et test. Ne touche jamais à Transported."""
    # Étape 1 : copie pour ne pas modifier le DataFrame d'origine.
    df = df.copy()

    # Étape 2 : découpe Cabin "B/0/P" en Deck / Numéro / Côté.
    # Le Deck encode la planète (vu en EDA : G=Earth, A/B/C=Europa).
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0].fillna("Missing")
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce").fillna(-1).astype(int)
    df["Side"] = cabin[2].fillna("Missing")

    # Étape 3 : découpe PassengerId "0003_02" en groupe + taille de groupe.
    # En groupe = 57% transporté vs solo = 45% (vu en EDA).
    df["GroupId"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("GroupId")["GroupId"].transform("size")
    df["Solo"] = df["GroupSize"] == 1

    # Étape 4 : résume les 5 dépenses (très zero-inflated : ~62% de zéros).
    # Règle métier vérifiée en 01 : CryoSleep=True => aucune dépense.
    df["TotalSpent"] = df[SPEND].fillna(0).sum(axis=1)
    df["NoSpending"] = (df[SPEND].fillna(0) == 0).all(axis=1)
    df[SPEND] = df[SPEND].fillna(0)

    # Étape 5 : convertit les booléens restés en object à cause des NaN (~200 manquants).
    df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(bool)
    df["VIP"] = df["VIP"].fillna(False).astype(bool)

    # Étape 6 : impute Age et catégories avec les stats du train (anti-fuite).
    df["Age"] = df["Age"].fillna(stats["age_median"])
    df["HomePlanet"] = df["HomePlanet"].fillna(stats["home_mode"])
    df["Destination"] = df["Destination"].fillna(stats["dest_mode"])

    # Étape 7 : extrait le nom de famille, puis supprime les colonnes brutes.
    # On garde les versions parsées (Deck, Side, GroupSize...) pour le modèle.
    df["Surname"] = df["Name"].str.split().str[-1].fillna("Missing")

    return df.drop(columns=["Cabin", "Name", "GroupId", "PassengerId"])


if __name__ == "__main__":
    # Étape 8 : applique exactement les mêmes règles au train et au test.
    print("Préparation des colonnes...")
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")

    stats = learn_stats(train_raw)
    print(f"stats: {stats}")
    train = make_features(train_raw, stats)
    test = make_features(test_raw, stats)

    print(f"train: {train_raw.shape} -> {train.shape}")
    print(f"test: {test_raw.shape} -> {test.shape}")
    print(train.head(3).to_string())
    print("Colonnes:", list(train.columns))

    # Vérif : après make_features il ne doit rester aucun NaN (tout est imputé).
    print("NaN restants train:", int(train.isna().sum().sum()))
    print("NaN restants test:", int(test.isna().sum().sum()))

    import missingno as msno

    # Avant features (train brut) : on voit les ~200 NaN par colonne.
    msno.matrix(train_raw, figsize=(12, 4))
    plt.title("NaN avant features (train brut)")
    plt.savefig(PLOTS_DIR / "02_msno_before.png", bbox_inches="tight")
    plt.close()
    print("Figure sauvée :", PLOTS_DIR / "02_msno_before.png")

    # Après features : normal de ne presque rien voir, tout est imputé.
    msno.matrix(train, figsize=(12, 4))
    plt.title("NaN après features (train traité)")
    plt.savefig(PLOTS_DIR / "02_msno_after.png", bbox_inches="tight")
    plt.close()
    print("Figure sauvée :", PLOTS_DIR / "02_msno_after.png")
