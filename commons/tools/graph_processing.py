from functools import partial

import networkx as nx
import pandas as pd
import tqdm

from commons.tools.basicFunctions import rand_iterator


# parallel computing config
# pool = Pool(cpu_count())

# extract sigma and omega index (A fork from networkx functions, parallel versions)
def small_world_index(G, pool, niter=100, nrand=10, seed=None):
    """Returns the small-world coefficient (sigma) of the given graph.

    The small-world coefficient is defined as:
    sigma = C/Cr / L/Lr
    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Cr and Lr are respectively the average
    clustering coefficient and average shortest path length of an equivalent
    random graph.

    A graph is commonly classified as small-world if sigma>1.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    niter : integer (optional, default=100)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.
    nrand : integer (optional, default=10)
        Number of random graphs generated to compute the average clustering
        coefficient (Cr) and average shortest path length (Lr).
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    sigma : float
        The small-world coefficient of G.

    Notes
    -----
    The implementation is adapted from Humphries et al. [1]_ [2]_.

    References
    ----------
    .. [1] The brainstem reticular formation is a small-world, not scale-free,
           network M. D. Humphries, K. Gurney and T. J. Prescott,
           Proc. Roy. Soc. B 2006 273, 503-511, doi:10.1098/rspb.2005.3354.
    .. [2] Humphries and Gurney (2008).
           "Network 'Small-World-Ness': A Quantitative Method for Determining
           Canonical Network Equivalence".
           PLoS One. 3 (4). PMID 18446219. doi:10.1371/journal.pone.0002051.
    """
    import numpy as np
    # Compute the mean clustering coefficient and average shortest path length
    # for an equivalent random graph
    rand_par = partial(rand_iterator, *[G, niter, seed])

    randMetrics = list(tqdm.tqdm(pool.imap(rand_par, list(range(nrand))), total=nrand))
    randMetrics = pd.DataFrame.from_dict(randMetrics)

    C = nx.transitivity(G)
    if nx.is_connected(G):
        L = nx.average_shortest_path_length(G)
    else:
        L = np.mean([nx.average_shortest_path_length(g) for g in nx.connected_component_subgraphs(G)])
    Cr = np.mean(randMetrics["C"])
    Cl = np.mean(randMetrics["Co"])
    Lr = np.mean(randMetrics["L"])

    sigma = (C / Cr) / (L / Lr)
    omega = (Lr / L) - (C / Cl)

    return sigma, omega

