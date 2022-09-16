import numpy as np
import matplotlib.pyplot as plt
import jax
import matplotlib as mpl
import wazy


AA_list = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "B",
    "Z",
    "X",
    "*",
]


def test(f, N, repeats, L, pretrain=True, key=0):
    key = jax.random.PRNGKey(key)
    np.random.seed(key)

    if pretrain:
        pre_results = do_boa(f, N, repeats, L, wazy.BOAlgorithm(), key)
        print("pretrain done", np.max(pre_results))
        pre_ei_results = do_boa(
            f,
            N,
            repeats,
            L,
            wazy.BOAlgorithm(alg_config=wazy.AlgConfig(bo_aq_fxn="ei")),
            key,
        )
        print("pretrain_ei done", np.max(pre_results))
    rand_results = do_rand(f, N, repeats, L)
    print("random done", np.max(rand_results))

    mcmc_results = do_boa(f, N, repeats, L, wazy.MCMCAlgorithm(L), key)
    print("mcmc done", np.max(mcmc_results))

    ohc_ucb_results = do_boa(
        f,
        N,
        repeats,
        L,
        wazy.BOAlgorithm(
            model_config=wazy.EnsembleBlockConfig(pretrained=False),
            alg_config=wazy.AlgConfig(bo_aq_fxn="ucb"),
        ),
        key,
    )
    print("ohc_ucb done", np.max(ohc_ucb_results))

    ohc_ei_results = do_boa(
        f,
        N,
        repeats,
        L,
        wazy.BOAlgorithm(
            model_config=wazy.EnsembleBlockConfig(pretrained=False),
            alg_config=wazy.AlgConfig(bo_aq_fxn="ei"),
        ),
        key,
    )
    print("ohc_ei done", np.max(ohc_ei_results))

    ohc_max_results = do_boa(
        f,
        N,
        repeats,
        L,
        wazy.BOAlgorithm(
            model_config=wazy.EnsembleBlockConfig(pretrained=False),
            alg_config=wazy.AlgConfig(bo_aq_fxn="max"),
        ),
        key,
    )
    print("ohc_max done", np.max(ohc_max_results))

    plt.plot(ohc_ucb_results, label="OH-UCB")
    plt.plot(ohc_ei_results, label="OH-EI")
    plt.plot(ohc_max_results, label="OH-MAX")
    plt.plot(mcmc_results, label="MCMC")
    plt.plot(rand_results, label="Random")
    if pretrain:
        plt.plot(pre_results, label="Pretrained")
        plt.plot(pre_ei_results, label="Pretrained")
    plt.legend()
    plt.show()

    def curbest(x):
        return [np.max(x[:i]) for i in range(1, len(x) + 1)]

    plt.figure()
    plt.plot(curbest(ohc_ucb_results), label="OH-UCB")
    plt.plot(curbest(ohc_ei_results), label="OH-EI")
    plt.plot(curbest(ohc_max_results), label="OH-MAX")
    plt.plot(curbest(mcmc_results), label="MCMC")
    plt.plot(curbest(rand_results), label="Random")
    if pretrain:
        plt.plot(curbest(pre_results), label="Pretrained")
        plt.plot(pre_ei_results, label="Pretrained EI")
    plt.legend()
    plt.show()


def do_rand(f, N, repeats, L):
    rand_results = [0 for i in range(N)]
    for _ in range(repeats):
        best = 0
        for i in range(N):
            s = "".join(np.random.choice(AA_list[:20], size=(L,)))
            y = f(s)
            best = max(y, best)
            rand_results[i] += y
    rand_results = [r / repeats for r in rand_results]
    return rand_results


def do_boa(f, N, repeats, L, boa, key):

    results = [0 for i in range(N)]
    for _ in range(repeats):
        start = "".join(np.random.choice(AA_list[:20], size=(L,)))
        boa.tell(key, start, f(start))
        best = 0

        for i in range(N):
            key, _ = jax.random.split(key)
            s, a = boa.ask(key)
            y = f(s)
            best = max(y, best)
            boa.tell(key, s, y)
            results[i] += y
        print(".", end="", flush=True)
    results = [r / repeats for r in results]
    return results
