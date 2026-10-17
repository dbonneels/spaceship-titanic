#! /home/dominique/spaceship-titanic/.venv/bin/python3
# comparer les modèles sur les données de validation.
#
# Lit models/eval.pkl (écrit par 03_train.py) : ce script N'ENTRAÎNE RIEN.
# Il analyse et affiche :
#   1. tableau OOF (validation) + test verrouillé : log-loss / AUC / accuracy
#   2. IC 95 % par bootstrap + test de McNemar : l'écart est-il significatif ?
#   3. sweep du seuil de décision (calé sur l'OOF, appliqué une seule fois au test)
#   4. fuite de groupe : le test verrouillé partage des groupes avec dev, alors que
#      le vrai test Kaggle n'en partage aucun (0/4277) -> mesure l'optimisme
#   5. graphiques -> plots/ : ROC, calibration, matrice de confusion, seuil

from pathlib import Path

import colorama
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # WSL sans écran : on sauve en PNG, plt.show() ne marche pas.
import matplotlib.pyplot as plt
from colorama import Fore, Style
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)

colorama.init(autoreset=True, wrap=True)

SEED = 42
N_BOOT = 2000  # ré-échantillonnages bootstrap pour l'IC de l'accuracy
EPS = 1e-15
BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

COLORS = {
    "logistic": "#8e7cc3",
    "hgb": "#e07b39",
    "catboost": "#2f7ed8",
    "lightgbm": "#4caf50",
    "xgboost": "#d64545",
    "blend": "#111111",
}


def header(title: str, color: str = Fore.CYAN) -> None:
    bar = "=" * 16
    line = f"{bar} {title} {bar}"
    print(color + Style.BRIGHT + "\n" + line + Style.RESET_ALL)
    print(color + "-" * len(line) + Style.RESET_ALL)


def info(msg: str) -> None:
    print(Fore.WHITE + msg + Style.RESET_ALL)


def step(msg: str) -> None:
    print(Fore.YELLOW + Style.BRIGHT + msg + Style.RESET_ALL)


def show(nom: str, m: dict, color: str = Fore.WHITE, ci: tuple = None) -> None:
    """Une ligne du tableau : log-loss, AUC, accuracy (seuil 0.5) et IC optionnel."""
    txt = (
        f"  {color}{nom:34s}{Style.RESET_ALL}"
        f" logloss={m['logloss']:.5f}  auc={m['auc']:.4f}  acc={m['acc']:.4f}"
    )
    if ci is not None:
        txt += f"  IC95=[{ci[0]:.4f}, {ci[1]:.4f}]"
    print(txt)


def metrics(y_true, proba) -> dict:
    """Log-loss, AUC et accuracy (seuil 0.5) d'un vecteur de probabilités."""
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(proba, dtype=float), EPS, 1 - EPS)
    return {
        "logloss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "acc": float(accuracy_score(y, p >= 0.5)),
    }


def combine(probas: dict, names, weights: dict) -> np.ndarray:
    """Combinaison linéaire des probas : somme(w_i * p_i)."""
    return np.sum([weights[name] * probas[name] for name in names], axis=0)


def ci_bootstrap(y_true, pred, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    """IC 95 % de l'accuracy par bootstrap : montre l'épaisseur du bruit."""
    y = np.asarray(y_true).astype(bool)
    ok = (np.asarray(pred).astype(bool) == y).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ok), size=(n_boot, len(ok)))
    accs = ok[idx].mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def mcnemar(y_true, pred_a, pred_b) -> tuple:
    """Test exact de McNemar entre deux prédicteurs sur le MÊME jeu."""
    y = np.asarray(y_true).astype(bool)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    a_seul = int(np.sum(a_ok & ~b_ok))  # A a raison, B a tort
    b_seul = int(np.sum(~a_ok & b_ok))  # A a tort, B a raison
    if a_seul + b_seul == 0:
        return 1.0, a_seul, b_seul
    p = binomtest(min(a_seul, b_seul), a_seul + b_seul, 0.5).pvalue
    return float(p), a_seul, b_seul


def plot_roc(axes, titre: str, y, store: dict, combo, weights) -> None:
    for name, p in store.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes.plot(
            fpr,
            tpr,
            color=COLORS.get(name, "gray"),
            lw=1.4,
            label=f"{name} (AUC={roc_auc_score(y, p):.4f})",
        )
    p_blend = combine(store, combo, weights)
    fpr, tpr, _ = roc_curve(y, p_blend)
    axes.plot(
        fpr,
        tpr,
        color=COLORS["blend"],
        lw=2.6,
        label=f"blend (AUC={roc_auc_score(y, p_blend):.4f})",
    )
    axes.plot([0, 1], [0, 1], "k--", lw=0.8)
    axes.set_title(f"ROC - {titre}")
    axes.set_xlabel("taux de faux positifs")
    axes.set_ylabel("taux de vrais positifs")
    axes.legend(fontsize=8, loc="lower right")


def main() -> None:
    path = MODELS_DIR / "eval.pkl"
    if not path.exists():
        raise SystemExit(f"{path} absent -> lance d'abord 03_train.py")

    # Étape 1 : chargement de l'artefact écrit par 03_train.py.
    header("ÉTAPE 1 - CHARGEMENT", Fore.BLUE)
    d = joblib.load(path)
    models = d["models"]
    combo = d["combo"]
    weights = d["weights"]
    y_dev, y_te = np.asarray(d["y_dev"]), np.asarray(d["y_test"])
    oof, te = d["oof"], d["test"]
    g_dev, g_te = np.asarray(d["group_dev"]), np.asarray(d["group_test"])
    blend_oof = combine(oof, combo, weights)
    blend_te = combine(te, combo, weights)
    info(f"protocole       : {d['protocol']}")
    info(f"combo verrouillé: {'+'.join(combo)}  (poids " + ", ".join(f"{n}={weights[n]:.3f}" for n in combo) + ")")
    info(f"dev (OOF)={len(y_dev)} lignes | test verrouillé={len(y_te)} lignes | {len(models)} modèles")

    # Étape 2 : comparaison sur la validation. Les folds sont groupés, donc chaque
    # ligne OOF est vue par un modèle qui n'a connu aucun membre de sa famille.
    header("ÉTAPE 2 - COMPARAISON SUR LA VALIDATION (OOF)", Fore.GREEN)
    m_oof = {name: metrics(y_dev, oof[name]) for name in models}
    best_single = min(models, key=lambda n: m_oof[n]["logloss"])
    for name in sorted(models, key=lambda n: m_oof[n]["logloss"]):
        show(name, m_oof[name], Fore.CYAN)
    show("BLEND " + "+".join(combo), metrics(y_dev, blend_oof), Fore.GREEN)
    info(f"→ meilleur modèle seul à l'OOF (log-loss) : {best_single}")

    # Étape 3 : mesure honnête, une seule fois, sur le test verrouillé.
    header("ÉTAPE 3 - TEST VERROUILLÉ (mesure honnête)", Fore.RED)
    m_te = {name: metrics(y_te, te[name]) for name in models}
    for name in sorted(models, key=lambda n: m_oof[n]["logloss"]):
        ci = ci_bootstrap(y_te, te[name] >= 0.5)
        show(name, m_te[name], Fore.CYAN, ci)
    ci_blend = ci_bootstrap(y_te, blend_te >= 0.5)
    show("BLEND " + "+".join(combo), metrics(y_te, blend_te), Fore.RED, ci_blend)
    info(f"largeur de l'IC 95 % du blend : ±{100 * (ci_blend[1] - ci_blend[0]) / 2:.2f} pts d'accuracy")
    info("→ tout écart plus petit que cette largeur n'est pas distinguable du hasard.")

    # Étape 4 : l'écart blend vs meilleur modèle seul est-il significatif ?
    header("ÉTAPE 4 - SIGNIFICATIVITÉ (McNemar)", Fore.MAGENTA)
    for jeu, y, p_blend, p_single in (
        ("OOF ", y_dev, blend_oof, oof[best_single]),
        ("test", y_te, blend_te, te[best_single]),
    ):
        p_val, a, b = mcnemar(y, p_blend >= 0.5, p_single >= 0.5)
        verdict = "NON significatif (les deux se valent)" if p_val > 0.05 else "significatif au seuil 5 %"
        info(
            f"  {jeu} : blend vs {best_single:9s} p={p_val:.3f} "
            f"(blend seul correct={a:3d}, {best_single} seul correct={b:3d}) -> {verdict}"
        )

    # Étape 5 : seuil de décision. Calé sur l'OOF, appliqué une seule fois au test.
    header("ÉTAPE 5 - SEUIL DE DÉCISION", Fore.MAGENTA)
    seuils = np.arange(0.30, 0.701, 0.01)
    accs_oof = np.array([accuracy_score(y_dev, blend_oof >= t) for t in seuils])
    t_best = float(seuils[int(np.argmax(accs_oof))])
    acc_oof_05 = accuracy_score(y_dev, blend_oof >= 0.5)
    acc_te_05 = accuracy_score(y_te, blend_te >= 0.5)
    acc_te_t = accuracy_score(y_te, blend_te >= t_best)
    info(f"  OOF  : acc={acc_oof_05:.4f} à 0.50 | max={accs_oof.max():.4f} au seuil {t_best:.2f}")
    info(f"  test : acc={acc_te_05:.4f} à 0.50 | {acc_te_t:.4f} au seuil {t_best:.2f} (seuil appris sur l'OOF)")
    info("  ⚠ la courbe est plate : le 'meilleur' seuil est lui aussi bruité, ne pas surinterpréter.")

    # Étape 6 : fuite de groupe. Point clé : le vrai test Kaggle ne partage AUCUN
    # groupe avec train (0/4277), alors que notre split interne en partage.
    header("ÉTAPE 6 - FUITE DE GROUPE DANS LE TEST VERROUILLÉ", Fore.YELLOW)
    vus = np.isin(g_te, np.unique(g_dev))
    info(f"  {int(vus.sum())}/{len(g_te)} passagers de test ({100 * vus.mean():.1f} %) ont un groupe DÉJÀ VU en dev")
    info("  dans le vrai jeu Kaggle : 0/4277 (aucune famille partagée entre train.csv et test.csv)")
    for nom, p in (("BLEND " + "+".join(combo), blend_te), (best_single, te[best_single])):
        for lbl, mask in (("groupe vu   ", vus), ("groupe inédit", ~vus)):
            acc = accuracy_score(y_te[mask], p[mask] >= 0.5)
            info(f"  {nom:26s} {lbl} : acc={acc:.4f} (n={int(mask.sum())})")
    info("  → ces 1739 lignes ne sont donc PAS un test parfaitement honnête :")
    info("    le vrai score de soumission sera plus bas, d'autant plus que l'écart ci-dessus est net.")

    # Étape 7 : graphiques.
    header("ÉTAPE 7 - GRAPHIQUES", Fore.CYAN)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_roc(axes[0], "OOF (validation)", y_dev, oof, combo, weights)
    plot_roc(axes[1], "test verrouillé", y_te, te, combo, weights)
    fig.tight_layout()
    p1 = PLOTS_DIR / "04_roc.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    info(f"  sauvé : {p1}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (titre, y, store) in zip(
        axes, (("OOF (validation)", y_dev, oof), ("test verrouillé", y_te, te))
    ):
        for nom, p in (
            ("blend " + "+".join(combo), combine(store, combo, weights)),
            (best_single, store[best_single]),
        ):
            prob_true, prob_pred = calibration_curve(
                y, np.clip(p, EPS, 1 - EPS), n_bins=10, strategy="quantile"
            )
            ax.plot(prob_pred, prob_true, "o-", lw=1.4, label=f"{nom} (logloss={log_loss(y, np.clip(p, EPS, 1 - EPS)):.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="parfaitement calibré")
        ax.set_title(f"Calibration - {titre}")
        ax.set_xlabel("proba prédite (moyenne par bin)")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("fréquence observée de Transported")
    fig.tight_layout()
    p2 = PLOTS_DIR / "04_calibration.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    info(f"  sauvé : {p2}")

    cm = confusion_matrix(y_te, blend_te >= 0.5)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13,
            )
    ax.set_xticks([0, 1], ["prédit False", "prédit True"])
    ax.set_yticks([0, 1], ["vrai False", "vrai True"])
    ax.set_title("Matrice de confusion - blend sur test")
    fig.tight_layout()
    p3 = PLOTS_DIR / "04_confusion.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    info(f"  sauvé : {p3}")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(seuils, accs_oof * 100, color="#2f7ed8", lw=1.8, label="accuracy OOF")
    ax.axvline(t_best, color="#d64545", ls="--", lw=1.2, label=f"meilleur seuil OOF = {t_best:.2f}")
    ax.axhline(acc_te_05 * 100, color="#111111", ls=":", lw=1.2, label=f"test à 0.50 = {acc_te_05 * 100:.2f} %")
    ax.set_xlabel("seuil de décision")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("L'accuracy selon le seuil (OOF) - le plateau montre le bruit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p4 = PLOTS_DIR / "04_threshold.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    info(f"  sauvé : {p4}")

    # Étape 8 : résumé à retenir.
    header("ÉTAPE 8 - RÉSUMÉ", Fore.CYAN)
    p_val, a, b = mcnemar(y_te, blend_te >= 0.5, te[best_single] >= 0.5)
    info(f"  meilleur modèle seul (OOF) : {best_single} (logloss {m_oof[best_single]['logloss']:.5f})")
    info(f"  blend retenu               : {'+'.join(combo)} (poids " + ", ".join(f"{n}={weights[n]:.3f}" for n in combo) + ")")
    info(f"  OOF   : logloss={metrics(y_dev, blend_oof)['logloss']:.5f}  auc={metrics(y_dev, blend_oof)['auc']:.4f}  acc={metrics(y_dev, blend_oof)['acc']:.4f}")
    info(f"  test  : logloss={metrics(y_te, blend_te)['logloss']:.5f}  auc={metrics(y_te, blend_te)['auc']:.4f}  acc={metrics(y_te, blend_te)['acc']:.4f}")
    info(f"  McNemar blend vs {best_single} (test) : p={p_val:.3f} -> {'non significatif' if p_val > 0.05 else 'significatif'}")
    info(f"  test contenant un groupe déjà vu : {int(vus.sum())}/{len(y_te)} (vrai Kaggle : 0)")
    info("  → le chiffre le plus fiable est le log-loss / AUC (métriques lisses) ;")
    info("    l'accuracy sur 1739 lignes est trop bruitée pour départager les modèles.")


if __name__ == "__main__":
    main()
