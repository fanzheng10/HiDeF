import os
import numpy as np
import pandas as pd
from scipy.sparse import *
from scipy.stats import *


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


def jaccard_matrix(matA, matB, threshold=0.75, return_mat=False):  # assume matA, matB are sorted
    '''
    Calculate jaccard matrix between all pairs between two sets of clusters.

    Parameters
    ----------
    matA : 2D array or scipy.sparse.csr_matrix
        axis 0 for clusters, axis 1 for nodes in network
    matB : 2D array or scipy.sparse.csr_matrix
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
    if not isinstance(matA, csr_matrix):
        matA = csr_matrix(matA)
    if not isinstance(matB, csr_matrix):
        matB = csr_matrix(matB)

    both = matA.dot(matB.T)

    either = (np.tile(matA.getnnz(axis=1), (matB.shape[0], 1)) + matB.getnnz(axis=1)[:, np.newaxis]).T - both
    jac = 1.0 * both / either
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


def containment_indices_boolean(A, B):  # TODO: test sparse matrix here
    '''
    Calculate a matrix of containment index for two lists of clusters

    Parameters
    ___________
    matA : 2D np.array
        axis 0 for clusters, axis 1 for nodes in network
    matB : 2D np.array

    Returns
    ________
    CI: 2D np.array
    '''
    count = np.count_nonzero(A, axis=1)

    A = A.astype(float)
    B = B.astype(float)
    overlap = A.dot(B.T)

    CI = overlap / count[:, None]
    return CI


def node2mat(f, g2ind, format='node', has_persistence=False):
    '''
    convert a text file to binary matrix (input of weaver)

    Parameters
    ----------
    f: str
        the input TSV file
    g2ind: dict
        a dictionary to index genes
    format: str
        accepted values are 'node' or 'clixo'
    has_persistence:
        if True, the input file has an extra column, here usually the persistence
    Returns
    --------
    data: a dictionary, contain the field 'cluster', 'name', and could contain 'extra.data'
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
        arr = np.zeros(n, )
        arr[gsi] = 1
        mat.append(arr)
    data = {'cluster': mat, 'name': df[0].tolist(), 'extra.data': None}
    if has_persistence:
        persistence = df[3].tolist()
        data['extra.data'] = persistence
    return data


def data2graph(datafile, outfile=None, k=15, snn=-1, mydist='cosine'):
    '''
    take a dataframe [n_samples x n_features] as input, and output knn or snn graph

    Parameters
    ----------
    datafile: str
        input tsv file
    outfile: str
        file name of output edge list
    k: int
        number of neighbors for each sample
    snn: float
        a threshold of Jaccard index if going to calculate shared nearest neighbor graph
    mydist: string or callable
        distance metric to calculate neighbors
    Returns
    --------
    idx: tuple of two numpy.array
        the indices of entries of the output matrix
    '''
    assert os.path.isfile(outfile) is False
    from sklearn.neighbors import kneighbors_graph
    data = pd.read_csv(datafile, sep='\t', index_col=0)
    data.fillna(0, inplace=True)
    data_npy = np.array(data)

    adj = kneighbors_graph(data_npy, n_neighbors=k, metric=mydist)  # note this is assymetric
    nodes = data.index.tolist()
    if snn > 0:
        idx = jaccard_matrix(adj, adj, threshold=snn)
    else:
        idx = np.where(adj.todense() > 0)
    if not (outfile is None):
        with open(outfile, 'w') as fh:
            for i in range(len(idx[0])):
                if (snn > 0) and (idx[0][i] > idx[1][i]):
                    continue
                name1, name2 = nodes[idx[0][i]], nodes[idx[1][i]]

                fh.write('{}\t{}\n'.format(name1, name2))
    return idx