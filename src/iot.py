"""Transport optimal inverse : Sinkhorn et apprentissage de la matrice d'interaction A (coût bilinéaire)."""

import warnings

import torch as t
from einops import einsum


@t.no_grad()  
def sinkhorn(C, mu, nu, lam=1.0, tol=1e-12, max_iter=10_000, a=None):
    """Plan entropique π(C) : minimise <π, C> - H(π)/lam sous π 1 = mu, π^T 1 = nu.

    Le plan s'écrit π_ij = exp(lam * (a_i + b_j - C_ij)) (plan entropique, Li et al., 2019).
    Sinkhorn alterne deux mises à jour exactes des potentiels :
      - b : les colonnes de π somment à nu ;
      - a : les lignes de π somment à mu.
    Tout est calculé en log (logsumexp) : exp(-lam * C) sous-déborde dès que
    lam * |C| dépasse quelques centaines, son logarithme jamais.

    Tenseurs en float64 : en float32, tol=1e-12 n'est jamais atteint.

    C        (m, n)  coût ; pour IOT bilinéaire, C = -U @ A @ V.T
    mu, nu   (m,), (n,)  marginales strictement positives, de même masse
    lam      λ > 0, inverse de la régularisation entropique
    tol      arrêt quand sum_i |(π 1)_i - mu_i| < tol (les colonnes sont exactes)
    a        potentiel des lignes de départ ; passer celui de l'appel précédent
             (démarrage à chaud) accélère la boucle d'apprentissage de A

    Renvoie π (m, n).
    """
    log_mu, log_nu = t.log(mu), t.log(nu)
    K = -lam * C                                   # log du noyau de Gibbs exp(-λC)
    a = t.zeros_like(mu) if a is None else a
    for _ in range(max_iter):
        # Colonnes : sum_i exp(λ(a_i + b_j - C_ij)) = nu_j, résolu en b_j.
        b = (log_nu - t.logsumexp(K + lam * a[:, None], dim=0)) / lam
        # s_i = log sum_j exp(λ(b_j - C_ij)), donc (π 1)_i = exp(λ a_i + s_i).
        s = t.logsumexp(K + lam * b[None, :], dim=1)
        if (t.exp(lam * a + s) - mu).abs().sum() < tol:
            break
        # Lignes : sum_j exp(λ(a_i + b_j - C_ij)) = mu_i, résolu en a_i.
        a = (log_mu - s) / lam
    else:
        warnings.warn(f"sinkhorn : pas de convergence en {max_iter} itérations")
    pi = t.exp(K + lam * (a[:, None] + b[None, :]))
    return pi

def entropie_croisee(p1, p2):
    """
    p1 (m, n) : matrice de couplage 
    p2 (m, n) : matrice de couplage sans 0
    Renvoie l'entropie -E_p1[log(pi2)]
    """
    p2_log = t.log(p2)
    return -einsum(p1, p2_log, "m n, m n -> ")


def IOT(pi_hat, U, V, n_it=1000, rho=1, lambd=1, s=0.5):
    """Algorithme IOT de Li et al. (2019) sous l'hypothèse C = -U @ A @ V.T :
    Prend en entrée : 
    U (m, p) : Vecteur des m types utilisateurs normalisés
    V (n, q) : Vecteur des n types des items normalisés
    pi_hat (m, n): Matrice de fréquence des types 
    lambd (1,) Pénalité phi(pi) = <C, pi> - 1/lambd * H(pi)
    rho (1,) Pénalité pour L(A) = entropie_croisée(pi_hat, pi) + rho * ||A||^2
    """
    p, q = U.shape[1], V.shape[1]
    A = t.zeros((p, q), dtype=U.dtype)
    loss = t.empty(n_it+1)
    grad = t.zeros((p, q))

    mu, nu = einsum(pi_hat, "i j -> i"), einsum(pi_hat, "i j -> j")

    for k in range(0, n_it+1):
        pi_sink = sinkhorn(- U @ (A @ V.T), mu, nu,  lam = lambd)
        loss[k] = entropie_croisee(pi_hat, pi_sink) + rho * (A**2).sum()
        grad =  -lambd * U.T @ (pi_hat - pi_sink) @ V + 2 * rho * A
        A = A - s * grad

    return A, loss, grad


if __name__ == "__main__":
    # Test sur un modèle bien spécifié : on tire des couples sous un A connu, puis on vérifie qu'IOT le retrouve.
    # Lancer depuis la racine : python src/iot.py
    import matplotlib.pyplot as plt               # outil local, pas dans requirements.txt

    t.manual_seed(0)
    m, n, p, q, N = 30, 40, 4, 3, 20_000          # types utilisateurs, types items, variables, couples tirés
    lambd, rho, s, n_it = 1.0, 0.0, 0.5, 300

    # Profils centrés réduits, marginales aléatoires, vrai A
    U, V = t.randn(m, p, dtype=t.float64), t.randn(n, q, dtype=t.float64)
    U, V = (U - U.mean(0)) / U.std(0), (V - V.mean(0)) / V.std(0)
    mu0, nu0 = t.rand(m, dtype=t.float64), t.rand(n, dtype=t.float64)
    mu0, nu0 = mu0 / mu0.sum(), nu0 / nu0.sum()
    A_vrai = 0.5 * t.randn(p, q, dtype=t.float64)

    # N couples tirés selon le vrai plan : pi_hat_ij = N_ij / N
    tirages = t.multinomial(sinkhorn(-U @ A_vrai @ V.T, mu0, nu0, lam=lambd).flatten(), N, replacement=True)
    pi_hat = t.bincount(tirages, minlength=m * n).reshape(m, n).double() / N

    A, loss, grad = IOT(pi_hat, U, V, n_it=n_it, rho=rho, lambd=lambd, s=s)

    # Références : perte du vrai A (mêmes marginales pi_hat que IOT) et entropie H0 de pi_hat, borne inférieure de la perte
    pi_vrai = sinkhorn(-U @ A_vrai @ V.T, pi_hat.sum(1), pi_hat.sum(0), lam=lambd)
    loss_vrai = entropie_croisee(pi_hat, pi_vrai) + rho * (A_vrai**2).sum()
    H0 = -t.xlogy(pi_hat, pi_hat).sum()

    for k in range(0, n_it + 1, n_it // 10):
        print(f"itération {k:4d}   perte {loss[k]:.6f}")
    print(f"perte du vrai A       : {loss_vrai:.6f}   (la perte finale doit être en dessous)")
    print(f"borne H0              : {H0:.6f}   (jamais atteinte avec des cases nulles)")
    print(f"perte décroissante    : {bool((loss.diff() <= 1e-5).all())}   (sinon, baisser s)")
    print(f"gradient max final    : {grad.abs().max():.1e}   (doit tendre vers 0 à l'optimum)")
    print(f"erreur relative sur A : {(A - A_vrai).norm() / A_vrai.norm():.3f}   (petite si rho = 0)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss, color="#2a78d6", lw=2)
    ax.axhline(loss_vrai, color="#52514e", lw=1, ls="--")
    ax.annotate("perte du vrai A", (n_it, loss_vrai), xytext=(0, 4), textcoords="offset points",
                ha="right", va="bottom", color="#52514e")
    ax.set(title=f"Perte d'IOT  (rho = {rho}, s = {s}, lambda = {lambd})", xlabel="itération", ylabel="perte")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.show()

