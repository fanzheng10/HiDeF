import numpy as np
import pandas as pd

def network_perturb(G, sample=0.8):
    '''
    perturb the network by randomly deleting some edges

    Parameters
    ----------
    G: input network
    sample: the fraction of edges to retain

    Returns
    ----------
    :return: the perturbed graph
    '''
    G1 = G.copy()
    edges_to_remove = [e.index for e in G1.es if np.random.rand() > sample]
    G1.delete_edges(edges_to_remove)
    return G1

def jaccard_matrix(matA, matB, threshold=0.75, return_mat=False): # assume matA, matB are sorted
    '''
    Calculate jaccard matrix between all pairs between two sets of clusters.

    Parameters
    ----------
    matA : scipy.sparse.csr_matrix
        axis 0 for clusters, axis 1 for nodes in network
    matB : scipy.sparse.csr_matrix
    threshold :
        a similarity cutoff for Jaccard index
    return_mat : bool
        set to true will also return the full pairwise Jaccard matrix

    Returns
    -------
    index : (np.array, np.array)
        two sets of indices; the cluster pairs implied by those indices satisfied threshold
    jac : np.array
        a full matrix of pairwise Jaccard indices
    '''
    both = matA.dot(matB.T)

    either = (np.tile(matA.getnnz(axis=1), (matB.shape[0],1)) + matB.getnnz(axis=1)[:, np.newaxis]).T -both
    jac = 1.0*both/either
    index = np.where(jac > threshold)
    if not return_mat:
        return index
    else:
        return index, jac

def containment_indices(A, B):
    from collections import defaultdict

    A = np.asarray(A)
    B = np.asarray(B)

    LA = np.unique(A)
    LB = np.unique(B)

    bA = np.zeros((len(LA), len(A)))
    for i, a in enumerate(LA):
        bA[i] = A == a

    bB = np.zeros((len(LB), len(B)))
    for i, b in enumerate(LB):
        bB[i] = B == b

    overlap = bA.dot(bB.T)
    count = bA.sum(axis=1)
    CI = overlap / count[:, None]

    return CI, LA, LB

def containment_indices_boolean(A, B):
    '''
    Calculate containment index between two clusters (how much B is contained in A)

    Parameters
    ----------
    A: np.array
        An binary array indicating the members of the cluster
    B: np.array
        Similar to A

    Returns
    --------
    CI: float
        the containment index
    '''`
    count = np.count_nonzero(A, axis=1)

    A = A.astype(float)
    B = B.astype(float)
    overlap = A.dot(B.T)

    CI = overlap / count[:, None]
    return CI

def containment_matrix(matA, matB):
    '''
    Calculate a matrix of containment index for two lists of clusters

    Parameters
    ___________
    matA : scipy.sparse.csr_matrix
        axis 0 for clusters, axis 1 for nodes in network
    matB : scipy.sparse.csr_matrix

    Returns
    ________
    contain: 2D np.array
    '''
    both = matA.dot(matB.T)
    countA = np.sum(matA, axis=1)
    contain = 1.0*both/np.tile(countA, (both.shape[1], 1)).T
    return contain


def node2mat(f, g2ind, format='node'):
    '''
    convert a text file to binary matrix (input of weaver)

    Parameters
    ----------
    f: str
        the input file
    g2ind: dict
        a dictionary to index genes
    format: str
        accepted values are 'node' or 'clixo'
    Returns
    --------
    mat: list of np.array
        a list of array representing cluster membership
    '''
    n = len(g2ind)
    df = pd.read_csv(f, sep='\t', header=None)
    mat = []
    for i, row in df.iterrows():
        if format == 'node':
            gs = row[2].split()
        elif format == 'clixo':
            gs = row[2].strip(',').split(',')
        else:
            raise ValueError('The format argument only accepts "node" or "clixo"')
        gsi = np.array([g2ind[g] for g in gs])
        arr = np.zeros(n,)
        arr[gsi] = 1
        mat.append(arr)
    return mat
