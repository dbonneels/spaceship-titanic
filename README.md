# Spaceship Titanic — pipeline ML de bout en bout

Prédire `Transported` sur la compétition Kaggle *Spaceship Titanic*, avec un protocole de
validation maison conçu pour ne pas se raconter d'histoires.

## Structure

| fichier | rôle |
|---|---|
| `01_eda.py` | analyse exploratoire, graphiques → `plots/` |
| `02_features.py` | parsing `Cabin` / `PassengerId`, imputations, features (`learn_stats` + `make_features`) |
| `03_train.py` | entraîne les 5 modèles, choisit le blend → `models/model.pkl`, `models/eval.pkl` |
| `04_evaluate.py` | analyse le blend (ne ré-entraîne rien, lit `models/eval.pkl`) → `plots/04_*.png` |
| `05_predict.py` | produit `submission.csv` depuis `models/model.pkl` → `plots/05_*.png` |
| `reconstruct_logistic.py` | pédagogie : reconstruit la fonction de décision de la LogisticRegression à la main |

Modèles comparés : `logistic`, `hgb`, `catboost`, `lightgbm`, `xgboost` (prétraitement
`OneHotEncoder` pour les catégorielles, `StandardScaler` en plus pour la logistique).

## Protocole de validation (V3)

1. **split 80 / 20** : `dev` (5 216 + 1 738 lignes) / `test` **verrouillé** (1 739 lignes).
   Graine et double split identiques à la V1 → le jeu test ne change pas d'une version à l'autre.
2. **sélection** : 5-fold `StratifiedGroupKFold` sur `dev`, donc les probabilités OOF (6 954 lignes)
   ne mélangent jamais deux membres d'une même famille. Objectif : **log-loss** (métrique lisse).
3. **26 combinaisons** de 2 à 5 modèles, puis **poids optimisés** au log-loss (≥ 0, somme = 1).
4. **mesure finale** : une seule fois sur le test verrouillé, jamais utilisée pour choisir.

Pourquoi ce protocole :

- une sélection à l'**accuracy** sur 1 738 lignes classait des écarts de 0.3 pt, soit ~0.3 SE :
  du bruit pur (le « gagnant » de la V1 était chanceux) ;
- un split **aléatoire** mettait ~30 % des familles à cheval entre le train et la validation
  des folds, ce qui gonflait les probas OOF de hgb et xgboost.

## Résultats

| | log-loss | AUC | accuracy |
|---|---|---|---|
| **blend `hgb`+`catboost` (0.19 / 0.81)** — OOF | **0.3830** | **0.9037** | 0.8129 |
| **blend** — test verrouillé | **0.3553** | **0.9187** | 0.8154 |
| catboost (meilleur modèle seul) — OOF | 0.3836 | 0.9037 | 0.8131 |
| hgb — OOF | 0.3944 | 0.8974 | 0.8077 |
| lightgbm — OOF | 0.3945 | 0.8976 | 0.8069 |
| xgboost — OOF | 0.4254 | 0.8911 | 0.7968 |
| logistic — OOF | 0.4357 | 0.8782 | 0.7890 |

Le blend gagne sur les métriques lisses (log-loss, AUC), mais **McNemar blend vs catboost sur le
test : p = 0.31** → les deux sont statistiquement indistinguables. L'IC 95 % de l'accuracy du
blend sur 1 739 lignes est de ±1.8 pt : tout écart plus petit est du bruit.

## ⚠️ Limite connue : le chiffre de test est optimiste

Le split `dev` / `test` est **aléatoire**, donc les familles y sont coupées :

- **714 / 1 739 (41 %)** des passagers du test verrouillé ont un membre de leur famille dans `dev` ;
- accuracy sur ces lignes : **85.3 %** — contre **78.9 %** sur celles dont la famille est inconnue ;
- or le **vrai** jeu Kaggle ne partage **aucun** groupe ni aucune cabin entre `train.csv` et
  `test.csv` (**0 / 4 277**, vérifié).

Le test interne est donc **plus facile que la compétition** : le modèle y retrouve des proches
déjà vus à l'entraînement. **L'estimation défendable du score réel est ~0.79, pas 0.8154.**

Nuance : « groupe vu » (100 % de familles) et « groupe inédit » (93 % de solos) n'ont pas la même
composition, donc l'écart de 6.4 pts mélange fuite et différence de population. Le sous-ensemble
« groupe inédit **et** famille » (n = 72) donne 77.8 %, ce qui pointe vers la fuite — mais n = 72,
c'est indicatif et non conclusif.

À noter : la *sélection* du blend est déjà propre (les folds OOF de la V3 sont groupés, contrôle
`assert` : 0 groupe partagé). Ce qui reste biaisé est **uniquement la mesure finale**.

**Correctif identifié, non implémenté (V4)** : rendre le split `dev` / `test` group-aware
(`GroupShuffleSplit` sur le `GroupId`) pour que le test verrouillé reproduise le vrai scénario.
Ça ne rapporterait aucun point de score — seulement un chiffre juste.

## Lancer

```bash
uv sync                          # installe l'environnement (Python 3.12, voir uv.lock)
uv run python 01_eda.py          # ou .venv/bin/python 01_eda.py
uv run python 02_features.py
uv run python 03_train.py        # ~3 min : 25 fits de CV + fits finaux
uv run python 04_evaluate.py     # instantané, nécessite models/eval.pkl (écrit par 03)
uv run python 05_predict.py      # instantané, écrit submission.csv
```

## Versionné / ignoré

- **versionné** : scripts, `data/*.csv` (entrées), `plots/*.png`, `uv.lock`, `pyproject.toml`
- **ignoré** (régénéré à chaque run) : `catboost_info/`, `models/*.pkl`, `submission.csv`,
  `__pycache__/`, `.venv`
