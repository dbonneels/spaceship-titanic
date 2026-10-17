#! /home/dominique/spaceship-titanic/.venv/bin/python3
# explorer les données et les valeurs manquantes.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # WSL sans écran : on sauve les figures en PNG au lieu de les afficher.
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print(f"train: {train.shape}")
print(f"test: {test.shape}")
print(train.head())
print(test.head())

print("\n1. Types (dtypes) :")
print(train.dtypes)

print("\n2. Valeurs distinctes (nunique) :")
print(train.nunique().sort_values(ascending=False))

print("\n3. Doublons PassengerId :")
print(f"train: {train.duplicated('PassengerId').sum()}")
print(f"test: {test.duplicated('PassengerId').sum()}")

print("\n4. Valeurs manquantes train vs test :")
# .sum() additionne chaque colonne. Comme True vaut 1 et False vaut 0, on obtient le nombre de valeurs manquantes par colonne.
print(
    pd.concat(  # axis=1 : on place les deux Series côte à côte (en colonnes), et non l’une sous l’autre.
        [train.isna().sum(), test.isna().sum()],
        axis=1,
        keys=[
            "train",
            "test",
        ],  # keys=["train", "test"] : donne un nom à chaque colonne du résultat.
    ).sort_values("train", ascending=False)
)
# NB : pandas aligne sur l’index (les noms de variables). Si une variable n’existe que d’un côté, l’autre case est NaN.

# Bar chart 1 : nombre de NaN par colonne (train).
nan_counts = train.isna().sum().sort_values(ascending=False)
nan_counts = nan_counts[nan_counts > 0]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(nan_counts.index, nan_counts.values)
ax.set_title("Nombre de NaN par colonne (train)")
ax.set_ylabel("Nombre de NaN")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "nan_counts.png")
plt.close(fig)
print("Figure sauvée :", PLOTS_DIR / "nan_counts.png")

# Bar chart 2 : pourcentage de NaN par colonne (train).
nan_pct = (train.isna().mean() * 100).sort_values(ascending=False)
nan_pct = nan_pct[nan_pct > 0]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(nan_pct.index, nan_pct.values)
ax.set_title("% de NaN par colonne (train)")
ax.set_ylabel("% de NaN")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "nan_pct.png")
plt.close(fig)
print("Figure sauvée :", PLOTS_DIR / "nan_pct.png")

# Heatmap : où se trouvent les NaN (1 = manquant, 0 = présent).
# Échantillon de 500 passagers pour garder la figure lisible.
sample_na = train.isna().sample(n=min(500, len(train)), random_state=42)
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(sample_na.values, aspect="auto", interpolation="nearest")
ax.set_title("Heatmap des NaN (échantillon 500 passagers, train)")
ax.set_xlabel("Colonnes")
ax.set_ylabel("Passagers (échantillon)")
ax.set_xticks(range(len(train.columns)))
ax.set_xticklabels(train.columns, rotation=45, ha="right")
fig.colorbar(im, ax=ax, label="1 = NaN")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "nan_heatmap.png")
plt.close(fig)
print("Figure sauvée :", PLOTS_DIR / "nan_heatmap.png")

print("\n5. Répartition de la cible :")
print(train["Transported"].value_counts(normalize=True))

CAT = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

print("\n6. Univarié — catégorielles :")
for col in CAT:
    print(f"\n-- {col} --")
    print(train[col].value_counts(dropna=False, normalize=True).head(10))

print(
    "\n6b. Univarié : Age + dépenses :"
)  # résume la distribution des variables numériques, mesure la part de zéros dans les dépenses, et mesure la part de valeurs manquantes.
print(
    train[["Age", *SPEND]].describe()
)  # *SPEND déplie la liste : ["Age", *SPEND] équivaut à ["Age", "RoomService", "FoodCourt", ...]
print("\n% de zéros (dépenses) :")
print((train[SPEND] == 0).mean().sort_values(ascending=False))
print("\n% de manquants Age/dépenses :")
print(train[["Age", *SPEND]].isna().mean().sort_values(ascending=False))

# distribution zero-inflated et à queue lourde.
"""
Ne pas utiliser la moyenne comme valeur d’imputation : elle ne représente aucun passager typique. Préfère la médiane (0), ou un 0 logique pour les passagers en CryoSleep.
Transformer avec np.log1p(x) pour comprimer la queue et rendre les variables exploitables par un modèle linéaire (comme ta régression logistique).
Créer des features :
python
  train["TotalSpend"] = train[SPEND].sum(axis=1)
  train["HasSpent"] = (train["TotalSpend"] > 0).astype(int)

"""

print("\n6c. Univarié : Cabin / PassengerId / Name :")
print("Exemples Cabin :", train["Cabin"].dropna().head(5).tolist())
print("Exemples PassengerId :", train["PassengerId"].head(5).tolist())
print("Exemples Name :", train["Name"].dropna().head(5).tolist())

print("\n7. Bivarié : catégorielles vs Transported :")
for col in CAT:
    print(f"\n-- {col} vs Transported --")
    print(
        train.groupby(col, dropna=False)["Transported"]
        .agg(n="size", rate="mean")
        .sort_values("rate", ascending=False)
    )

print("\n7b. Bivarié — Age/dépenses vs Transported :")
train["TotalSpent"] = train[SPEND].sum(axis=1, skipna=False)
train["NoSpending"] = (train[SPEND].fillna(0) == 0).all(axis=1)
print(train.groupby("Transported")[["Age", "TotalSpent"]].median())
print("\nNoSpending vs Transported :")
print(train.groupby("NoSpending")["Transported"].agg(n="size", rate="mean"))
print("\nCryoSleep vs NoSpending (vérif règle métier) :")
print(pd.crosstab(train["CryoSleep"], train["NoSpending"], normalize="index"))

print("\n8. Plurivarié — Cabin Deck/Side, Groupe, interactions :")
train["Deck"] = train["Cabin"].str.split("/").str[0]
train["Side"] = train["Cabin"].str.split("/").str[2]
train["GroupId"] = train["PassengerId"].str.split("_").str[0]
train["GroupSize"] = train.groupby("GroupId")["GroupId"].transform("size")
train["Solo"] = train["GroupSize"] == 1

print("\n-- Deck vs Transported --")
print(
    train.groupby("Deck", dropna=False)["Transported"]
    .agg(n="size", rate="mean")
    .sort_values("rate", ascending=False)
)
print("\n-- Side vs Transported --")
print(train.groupby("Side", dropna=False)["Transported"].agg(n="size", rate="mean"))
print("\n-- Deck x HomePlanet (le Deck encode la planète ?) --")
print(pd.crosstab(train["Deck"], train["HomePlanet"], normalize="index").round(2))
print("\n-- Solo vs Transported --")
print(train.groupby("Solo")["Transported"].agg(n="size", rate="mean"))
print("\n-- HomePlanet x Destination vs Transported --")
print(
    train.groupby(["HomePlanet", "Destination"], dropna=False)["Transported"]
    .agg(n="size", rate="mean")
    .sort_values("rate", ascending=False)
    .head(15)
)

print("\n-- GroupSize vs Transported --")

for col in CAT:
    print(f"\n-- {col} vs Transported --")
    print(train.groupby(col, dropna=False)["Transported"].agg(n="size", rate="mean"))
