"""IOT lisible : types d'utilisateurs (k-means sur les genres regardés) x genres de films.

Deux tableaux de couplages K x L sont appris avec iot.IOT, U = V = identité :
  - « vus »   : N_kl = nombre de notes des utilisateurs du type k sur les films du genre l ;
  - « aimés » : même comptage, notes >= 4 seulement.
Un film à g genres compte 1/g dans chacun. Les marges sont la taille des types et des genres ;
A_vus dit qui regarde quoi, A_aimés qui aime quoi, et A_aimés - A_vus la part de bonnes notes d'un type
sur un genre, nette de la sévérité du type et de la réputation du genre : c'est la variable de goût du modèle de notes
  note = moyenne + biais utilisateur + biais film + beta * goût[type(u), genres(i)].

generate.py appelle completer(table) : le modèle retenu, K = 3.
Lancer depuis la racine : python src/types_iot.py            (K, rho choisis en validation interne)
"""

import os
import sys

import numpy as np
import torch as t
from sklearn.cluster import KMeans

from iot import IOT

SEED, RHO, N_IT = 0, 1e-4, 300
GENRES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genres.npz")


def construire_genres(chemin="data/namesngenre2.npy"):
    """Extrait les genres MovieLens de namesngenre2.npy dans src/genres.npz, pour que generate.py n'ait pas besoin de data/."""
    ng = np.load(chemin)
    noms = sorted({g for s in ng[:, 1] for g in s.split("|")} - {"(no genres listed)"})
    G = np.array([[g in s.split("|") for g in noms] for s in ng[:, 1]])
    np.savez_compressed(GENRES, G=G, noms=np.array(noms))


def matrice_genres():
    """Matrice films x genres W (chaque ligne somme à 1 ; 0 si le film n'a pas de genre) et noms des genres."""
    f = np.load(GENRES)
    W = f["G"].astype(float)
    return W / np.maximum(W.sum(1, keepdims=True), 1), list(f["noms"])


def charger():
    """Notes train/test et genres."""
    W, genres = matrice_genres()
    return np.load("data/ratings_train.npy"), np.load("data/ratings_test.npy"), W, genres


def types_utilisateurs(vu, W, K):
    """k-means sur la part de chaque genre dans les films notés par l'utilisateur. Renvoie le type de chacun."""
    P = vu @ W
    P /= np.maximum(P.sum(1, keepdims=True), 1)
    return KMeans(K, n_init=10, random_state=SEED).fit_predict(P)


def tableau(masque, type_u, W, K):
    """N_kl : masse des notes (masque) entre le type d'utilisateurs k et le genre l."""
    T = np.eye(K)[type_u]                              # utilisateurs x types, indicatrice
    return T.T @ (masque @ W)


def apprendre(N, rho=RHO):
    """IOT sur un tableau de comptages, U = V = identité : A_kl est l'affinité entre le type k et le genre l."""
    pi_hat = t.tensor(N / N.sum())
    K, L = pi_hat.shape
    s = 1 / pi_hat.max().item()                        # pas ~ inverse de la courbure (la plus grosse case)
    A, loss, _ = IOT(pi_hat, t.eye(K, dtype=t.float64), t.eye(L, dtype=t.float64), n_it=N_IT, rho=rho, s=s)
    assert (loss.diff() <= 1e-9).all(), "perte non décroissante : baisser s"
    return A.numpy()


def modele_notes(u, i, r, x, m, n, reg=2.0, n_sweep=15):
    """note = mu + b_u + b_i + beta x, moindres carrés alternés avec rétrécissement (ridge) des biais."""
    mu, bu, bi, beta = r.mean(), np.zeros(m), np.zeros(n), 0.0
    for _ in range(n_sweep):
        bu = np.bincount(u, r - mu - bi[i] - beta * x, m) / (np.bincount(u, minlength=m) + reg)
        bi = np.bincount(i, r - mu - bu[u] - beta * x, n) / (np.bincount(i, minlength=n) + reg)
        res = r - mu - bu[u] - bi[i]
        beta = (res * x).sum() / (x * x).sum() if x.any() else 0.0

    def predire(u_, i_, x_):
        return mu + bu[u_] + bi[i_] + beta * x_
    predire.mu, predire.bu, predire.bi = mu, bu, bi    # paramètres appris, accessibles après l'ajustement
    return predire, beta


def experience(R_app, R_eval, W, K):
    """Apprend types, tableaux, A et modèle de notes sur R_app ; renvoie rmse/accuracy sur R_eval et les cartes."""
    m, n = R_app.shape
    vu, aime = ~np.isnan(R_app), R_app >= 4
    type_u = types_utilisateurs(vu, W, K)
    N_vu, N_aime = tableau(vu, type_u, W, K), tableau(aime, type_u, W, K)
    A_vu, A_aime = apprendre(N_vu), apprendre(N_aime)
    gout = A_aime - A_vu                               # types x genres

    u, i = np.nonzero(vu); r = R_app[u, i]
    x = (gout[type_u[u]] * W[i]).sum(1)                # goût du type de u pour les genres de i
    predire, beta = modele_notes(u, i, r, x, m, n)
    biais, _ = modele_notes(u, i, r, np.zeros_like(x), m, n)

    qu, qi = np.nonzero(~np.isnan(R_eval)); qr = R_eval[qu, qi]
    qx = (gout[type_u[qu]] * W[qi]).sum(1)
    scores = {nom: score(p, qr) for nom, p in [("biais", biais(qu, qi, qx)), ("biais + goût", predire(qu, qi, qx))]}
    return scores, beta, dict(type_u=type_u, N_vu=N_vu, A_vu=A_vu, A_aime=A_aime, gout=gout)


def completer(table, K=3):
    """Complète toute la matrice de notes avec le modèle retenu, appris sur les notes présentes ; utilisé par generate.py."""
    t.manual_seed(SEED)
    W, _ = matrice_genres()
    m, n = table.shape
    vu, aime = ~np.isnan(table), table >= 4
    if W.shape[0] == n:
        type_u = types_utilisateurs(vu, W, K)
        gout = apprendre(tableau(aime, type_u, W, K)) - apprendre(tableau(vu, type_u, W, K))
        X = gout[type_u] @ W.T                         # goût de chaque utilisateur pour chaque film, m x n
    else:                                              # genres inconnus pour ces colonnes : biais seuls
        X = np.zeros((m, n))
    u, i = np.nonzero(vu)
    predire, _ = modele_notes(u, i, table[u, i], X[u, i], m, n)
    uu, ii = np.indices((m, n))
    pred = np.clip(predire(uu, ii, X), 0.5, 5)
    return np.where(vu, table, pred)


def score(pred, r):
    """Comme evaluate.py : RMSE, et accuracy après arrondi au demi-point."""
    pred = np.clip(pred, 0.5, 5)
    return np.sqrt(np.mean((pred - r) ** 2)), np.mean(np.clip(np.round(pred * 2) / 2, 0.5, 5) == r)


def grille(R_app, R_val, W, Ks=(2, 3, 4, 6), rhos=(1e-5, 1e-4, 1e-3, 1e-2), seuils=(3.5, 4.0, 4.5), regs=(1, 2, 5, 10)):
    """RMSE de validation de toutes les combinaisons (K, rho, seuil « aimé », ridge des biais), triées."""
    m, n = R_app.shape
    vu = ~np.isnan(R_app)
    u, i = np.nonzero(vu); r = R_app[u, i]
    qu, qi = np.nonzero(~np.isnan(R_val)); qr = R_val[qu, qi]
    res = []
    for K in Ks:
        type_u = types_utilisateurs(vu, W, K)
        N_vu = tableau(vu, type_u, W, K)
        for rho in rhos:
            A_vu = apprendre(N_vu, rho)
            for seuil in seuils:
                gout = apprendre(tableau(R_app >= seuil, type_u, W, K), rho) - A_vu
                x, qx = (gout[type_u[u]] * W[i]).sum(1), (gout[type_u[qu]] * W[qi]).sum(1)
                for reg in regs:
                    predire, _ = modele_notes(u, i, r, x, m, n, reg=reg)
                    res.append((score(predire(qu, qi, qx), qr)[0], K, rho, seuil, reg))
    return sorted(res)


def validation_interne(train, frac=0.2):
    """Coupe les notes du train en deux matrices : apprentissage (80 %) et validation (20 %)."""
    u, i = np.nonzero(~np.isnan(train))
    val = np.random.default_rng(SEED).random(len(u)) < frac
    R_app, R_val = train.copy(), np.full(train.shape, np.nan)
    R_app[u[val], i[val]], R_val[u[val], i[val]] = np.nan, train[u[val], i[val]]
    return R_app, R_val


def noms_des_types(res, genres):
    """Nomme chaque type par les deux genres qu'il regarde le plus par rapport aux autres types (A_vus)."""
    noms = []
    for k, (a, n_k) in enumerate(zip(res["A_vu"], res["N_vu"].sum(1))):
        top = np.argsort(-a)[:2]
        noms.append(f"{k} {genres[top[0]]}/{genres[top[1]]} ({np.bincount(res['type_u'])[k]} util.)")
    return noms


def cartes(res, genres, chemin):
    """Trois cartes de chaleur : qui regarde quoi, qui aime quoi, et la différence (le goût)."""
    import matplotlib.pyplot as plt                   # outil local, pas dans requirements.txt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 2.2 + 0.4 * len(res["A_vu"]) * 3))
    titres = ["A_vus : qui regarde quoi", "A_aimés : qui aime quoi (notes >= 4)",
              "A_aimés - A_vus : le goût, net de la sévérité du groupe et de la réputation du genre"]
    for ax, M, titre in zip(axes, [res["A_vu"], res["A_aime"], res["gout"]], titres):
        v = np.abs(M).max()
        im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
        ax.set(title=titre, xticks=range(len(genres)), yticks=range(len(M)))
        ax.set_xticklabels(genres, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(noms_des_types(res, genres), fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)
    print(f"cartes enregistrées : {chemin}")


if __name__ == "__main__":
    t.manual_seed(SEED)
    train, test, W, genres = charger()
    R_app, R_val = validation_interne(train)

    print("== validation interne (RMSE) : grille sur K, rho, seuil « aimé », ridge des biais")
    res = grille(R_app, R_val, W)
    for rmse, *config in res[:5]:
        print(f"  {rmse:.4f}  K={config[0]}  rho={config[1]:g}  seuil={config[2]}  ridge={config[3]}")
    for nom, j, vals in [("K", 1, (2, 3, 4, 6)), ("rho", 2, (1e-5, 1e-4, 1e-3, 1e-2)), ("seuil", 3, (3.5, 4.0, 4.5)), ("ridge", 4, (1, 2, 5, 10))]:
        print(f"  meilleure RMSE par {nom:5s} :", "  ".join(f"{v:g}: {min(row[0] for row in res if row[j] == v):.4f}" for v in vals))
    retenu = [row for row in res if row[1:] == (3, RHO, 4.0, 2)][0]
    print(f"  retenu K=3, rho={RHO:g}, seuil=4, ridge=2 : {retenu[0]:.4f} (écart au meilleur {retenu[0] - res[0][0]:.4f}, sous le bruit)")
    K = 3

    print(f"\n== test, K={K} (apprentissage sur tout ratings_train)")
    scores, beta, res = experience(train, test, W, K)
    for nom, (rmse, acc) in scores.items():
        print(f"  {nom:14s}: rmse {rmse:.4f}  accuracy {acc:.2%}")
    print(f"  beta {beta:+.2f}")

    print("\n== types (part des notes >= 4, puis genres les plus regardés et les plus aimés par rapport aux autres types)")
    N_aime = res["N_vu"] * 0 + tableau(train >= 4, res["type_u"], W, K)
    for k, nom in enumerate(noms_des_types(res, genres)):
        aime = ", ".join(genres[l] for l in np.argsort(-res["gout"][k])[:3])
        print(f"  {nom:40s} aime {N_aime[k].sum() / res['N_vu'][k].sum():.0%}   goût : {aime}")
    cartes(res, genres, sys.argv[1] if len(sys.argv) > 1 else "types_iot.png")
