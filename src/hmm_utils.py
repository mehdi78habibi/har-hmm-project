import numpy as np
from scipy.stats import multivariate_normal

def brute_force_p_observation(O, pi, A, means, covs):
    T, D = O.shape
    N = pi.shape[0]
    def bj(j, ot):
        return multivariate_normal.pdf(ot, mean=means[j], cov=covs[j], allow_singular=True)
    total = 0.0
    import itertools
    for path in itertools.product(range(N), repeat=T):
        p = pi[path[0]] * bj(path[0], O[0])
        for t in range(1, T):
            p *= A[path[t-1], path[t]] * bj(path[t], O[t])
        total += p
    return total

def forward_p_observation(O, pi, A, means, covs, log=False):
    T, D = O.shape
    N = pi.shape[0]
    B = np.zeros((T, N))
    for t in range(T):
        for j in range(N):
            B[t, j] = multivariate_normal.pdf(O[t], mean=means[j], cov=covs[j], allow_singular=True)
    alpha = pi * B[0]
    c = np.zeros(T)
    c[0] = alpha.sum()
    if c[0] == 0:
        return -np.inf if log else 0.0
    alpha = alpha / c[0]
    for t in range(1, T):
        alpha = (alpha @ A) * B[t]
        c[t] = alpha.sum()
        if c[t] == 0:
            return -np.inf if log else 0.0
        alpha = alpha / c[t]
    log_p = -np.sum(np.log(c))
    return log_p if log else float(np.exp(log_p))

def mk_tiny_hmm(seed=0):
    """
    یک HMM خیلی کوچک برای دمو:
    N=2 حالت، D=1 بُعد، دنباله با طول T=4
    """
    rng = np.random.default_rng(seed)
    N, D, T = 2, 1, 4
    pi = np.array([0.6, 0.4])
    A  = np.array([[0.8, 0.2],
                   [0.3, 0.7]])
    means = np.array([[0.0],   # حالت 0
                      [3.0]])  # حالت 1
    covs  = np.array([[[1.0]], [[1.0]]])  # کوواریانس‌های قطری
    # یک دنباله‌ی کوتاه مطابق با میانگین‌ها
    O = np.array([rng.normal(0,1),
                  rng.normal(0,1),
                  rng.normal(3,1),
                  rng.normal(3,1)]).reshape(T, D)
    return O, pi, A, means, covs
