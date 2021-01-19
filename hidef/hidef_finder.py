#! /usr/bin/env python

import networkx as nx
import igraph as ig
import argparse
import os, time
import pickle
import numpy as np
import pandas as pd
import scipy as sp

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import louvain
import leidenalg

from hidef import weaver
from hidef.utils import *
from hidef import LOGGER
import multiprocessing as mp


class Cluster(object):
    """
    The base class representing a cluster in hidef.

    Parameters
    ----------
    binary : np.array
        a binary vector indicating which objects belong to this cluster
    gamma : float
        the resolution parameter which generated this cluster
    """

    __slots__ = ['size',
                 'members',
                 'binary',
                 'padded',
                 'resolution_parameter']

    def __init__(self, binary, gamma):
        self.binary = np.squeeze(np.asarray(binary.todense()))
        self.members = np.where(self.binary)[0]
        self.size = len(self.members)
        self.resolution_parameter = '{:.4f}'.format(gamma)
        self.padded = False
        # self.index=None

    def calculate_similarity(self, cluster2):
        '''
        calculate the Jaccard similarity between two clusters

        Parameters
        ----------
        cluster2 : hidef_finder.Cluster

        Returns
        ----------
        : float
        '''
        # jaccard index
        arr1, arr2 = self.binary, cluster2.binary
        both = np.dot(arr1, arr2)
        either = len(arr1) - np.dot(1 - arr1, 1 - arr2)
        return 1.0 * both / either


class ClusterGraph(nx.Graph):  # inherit networkx digraph
    '''
    Extending nx.Graph class, each node is a hidef_finder.Cluster object
    '''

    def add_clusters(self, resolution_graph, new_resolution):
        '''
        Add new clusters to cluster graph once a new resolution is finished by the CD algorithm

        Parameters
        ----------
        resolution_graph: nx.Graph
            a graph in which nodes represent resolutions in the scan, and edges connecting the resolutions that are considered 'neighbors'
        new_resolution: float
            the resolution just visited by the CD algorithm

        Returns
        ----------
        '''
        resname_new = '{:.4f}'.format(new_resolution)
        new_clusters = []
        new_mat = resolution_graph.nodes[resname_new]['matrix']

        for i in range(new_mat.shape[0]):  # c is a list of node indices
            clu = Cluster(new_mat[i, :], new_resolution)
            # cluG.add_cluster(clu)
            new_clusters.append(clu)

        # compare c with all existing clusters in clusterGraph (maybe optimized in future)
        if len(self.nodes()) == 0:
            newnode = np.arange(0, len(new_clusters))
        else:
            newnode = np.arange(max(self.nodes()) + 1, max(self.nodes()) + 1 + len(new_clusters))
        resolution_graph.nodes[resname_new]['node_indices'] = newnode

        # compare against all other resolutions within range
        newedges = []
        for r in resolution_graph.neighbors(resname_new):
            if not ('node_indices' in resolution_graph.nodes[r]):  # not registered
                continue
            r_node_ids = resolution_graph.nodes[r]['node_indices']
            rmat = resolution_graph.nodes[r]['matrix']
            id_new, id_r = jaccard_matrix(new_mat, rmat, self.graph['sim_threshold'])
            if len(id_new) > 0:
                id_new = newnode[id_new]
                id_r = r_node_ids[id_r]
                for i in range(len(id_new)):
                    newedges.append((id_new[i], id_r[i]))

        self.add_edges_from(newedges)

        for nci in range(len(new_clusters)):
            nc = new_clusters[nci]
            ni = newnode[nci]
            if not ni in self.nodes:
                self.add_node(ni)
            self.nodes[ni]['data'] = nc  # is it a pointer?
            # self.nodes[ni]['data'].index = ni


def run_alg(Gs, alg, gamma=1.0, sample=1.0, layer_weights=None):
    '''
    Run community detection algorithm with a resolution parameter. Right now only use RB in Louvain/Leiden

    Parameters
    ----------
    Gs : a list of igraph.Graph
    alg : str
        choose between 'louvain' and 'leiden'
    gamma : float
        resolution parameter
    sample : if smaller than 1, randomly delete a fraction of edges each time
    layer_weights: a list of float
        specifying layer weights in the multilayer setting
    Returns
    ------
    C: scipy.sparse.csr_matrix
        a matrix recording the membership of each cluster

    '''
    if len(Gs) == 1:
        G = Gs[0]
        G1 = G.copy()
        if sample < 1:
            G1 = network_perturb(G, sample)
        if alg == 'louvain':
            partition_type = louvain.RBConfigurationVertexPartition
            partition = louvain.find_partition(G1, partition_type, resolution_parameter=gamma)
        elif alg == 'leiden':
            partition_type = leidenalg.RBConfigurationVertexPartition
            partition = leidenalg.find_partition(G1, partition_type, resolution_parameter=gamma)
        partitions = [partition]
    else:  # multiplex mode
        if layer_weights == None:
            layer_weights = [1.0 for _ in Gs]
        assert len(layer_weights) == len(Gs), 'layer weights inconsistent with the number of input networks'
        Gs1 = [G.copy() for G in Gs]
        if sample < 1:
            Gs1 = [network_perturb(G, sample) for G in Gs]
        if alg == 'louvain':
            partition_type = louvain.RBConfigurationVertexPartition
            optimiser = louvain.Optimiser()
            partitions = [partition_type(G, resolution_parameter=gamma) for G in Gs1]
            _ = optimiser.optimise_partition_multiplex(partitions, layer_weights=layer_weights)
        elif alg == 'leiden':
            partition_type = leidenalg.RBConfigurationVertexPartition
            # partition = leidenalg.find_partition_multiplex(Gs1, partition_type, resolution_parameter=gamma,
            #                                                layer_weights=layer_weights)
            optimiser = leidenalg.Optimiser()
            partitions = [partition_type(G, resolution_parameter=gamma) for G in Gs1]
            _ = optimiser.optimise_partition_multiplex(partitions, n_iterations=-1, layer_weights=layer_weights) # -1 means iterate until no further optimization
            # print([len(p) for p in partitions]) # debug


    # partition = sorted(partition, key=len, reverse=True)
    LOGGER.info('Resolution: {:.4f}; find {} clusters'.format(gamma, len(partitions[0])))

    return partition_to_membership_matrix(partitions[0])


def partition_to_membership_matrix(partition, minsize=4):
    '''

    Parameters
    ----------
    partition: class partition in the louvain-igraph package
    minsize: int
        minimum size of clusters; smaller clusters will be deleted afterwards

    Returns
    ----------
    C: scipy.sparse.csr_matrix
        a matrix recording the membership of each cluster
    '''
    clusters = sorted([p for p in partition if len(p) >= minsize], key=len, reverse=True)
    row, col = [], []
    for i in range(len(clusters)):
        row.extend([i for _ in clusters[i]])
        col.extend([x for x in clusters[i]])
    row = np.array(row)
    col = np.array(col)
    data = np.ones_like(row, dtype=int)
    C = sp.sparse.coo_matrix((data, (row, col)),
                             shape=(len(clusters), partition.n))  # TODO: make it not dependent on partition.n
    C = C.tocsr()
    return C


def update_resolution_graph(G, new_resolution, neighborhood_size, neighbor_density_threshold):
    '''
    Update the "resolution graph", which connect resolutions that are close enough

    Parameters
    ----------
    G: nx.Graph
        the "resolution graph"
    new_resolution: float
        the resolution just visited by the CD algorithm
    neighborhood_size: float
      if two resolutions (log-scale) differs smaller than this value, they are called 'neighbors'
    neighbor_density_threshold: int
        if a resolution has neighbors more than this number, it is called "padded". No more sampling will happen between two padded resolutions

    Returns
    ----------
    '''
    nodename = '{:.4f}'.format(new_resolution)
    G.add_node(nodename, resolution=new_resolution,
               padded=False)
    for v, vd in G.nodes(data=True):
        if v == nodename:
            continue
        if abs(np.log10(vd['resolution']) - np.log10(new_resolution)) < neighborhood_size:
            G.add_edge(v, nodename)
    newly_padded = []
    for v, deg in G.degree():
        if deg > neighbor_density_threshold:
            if not G.nodes[v]['padded']:
                newly_padded.append(v)
            G.nodes[v]['padded'] = True
    return newly_padded


def collapse_cluster_graph(cluG, components, p=100):
    '''

    take the cluster graph and collapse each component based on some consensus metric

    Parameters
    ----------
    cluG: hidef_finder.ClusterGraph
    components: list of list of int
        elements of the inner list are numbers to query a cluster in the cluster graph
    p: int or float
        remove nodes if they did not appear in more than t percent of clusters in one component (the p parameter in the paper)

    Returns
    ----------
    collapsed_clusters: a list of np.array
        binary numpy arrays indicating the members of each cluster after filtering by consensus

    See also
    ---------
    consensus
    '''
    collapsed_clusters = []
    for component in components:
        clusters = [cluG.nodes[v]['data'].binary for v in component]
        mat = np.stack(clusters)
        participate_index = np.mean(mat, axis=0)
        threshold_met = participate_index * 100 > p
        threshold_met = threshold_met.astype(int)
        collapsed_clusters.append(threshold_met)
    return collapsed_clusters


def run(Gs,
        jaccard=0.75,
        sample=0.9,
        minres=0.01,
        maxres=10,
        alg='leiden',
        maxn=None,
        density=0.1,
        neighbors=10,
        numthreads=mp.cpu_count(),
        layer_weights=None):
    '''
    Main function to run the Finder program

    Parameters
    ----------
    Gs: a list of igraph.Graph
        input network(s)
    jaccard: float
        use (0.5-1.0); a cutoff to call two clusters similar; the 'tau' parameter in paper
    sample: float
        parameter to perturb input network in each run by deleting edges; lower values delete more
    minres: float
        minimum resolution parameter
    maxres: float
        maximum resolution parameter
    maxn: float
        will explore resolution parameter until cluster number is similar to this number; will override 'maxres'
    alg: str
        can choose between 'louvain' or 'leiden'
    density: float
        inversed density of sampling resolution parameter. Use a smaller value to increase sample density (will increase running time)
    neighbors: int
        also affect sampling density; a larger value may have additional benefits of stabilizing clustering results
    bisect: deprecated
    numthreads: int
        Number of threads to run in parallel. Default is set to number of cores.

    Returns
    ----------
    '''
    min_diff_bisect_value = 1
    min_diff_resolution = 0.001

    # G = ig.Graph.Read_Ncol(G)
    for i in range(len(Gs)):
        Gs[i].simplify(multiple=False)  # remove self loop but keep weight
    cluG = ClusterGraph()
    cluG.graph['sim_threshold'] = jaccard
    cluG.graph['num_leaves'] = len(Gs[0].vs)

    resolution_graph = nx.Graph()

    # perform two initial louvain
    # minres_partition = run_alg(G, alg, minres)

    LOGGER.timeit('_resrange')
    LOGGER.info('Finding maximum resolution...')
    if maxn != None:
        next = True
        last_move = ''
        last_res = maxres
        maximum_while_loop = 20
        n_loop = 0
        while next == True:
            test_partition = run_alg(Gs, alg, maxres, layer_weights=layer_weights)
            n_loop += 1
            if n_loop > maximum_while_loop:
                LOGGER.warning(
                    'Reach upper limit of initial resolution searching, cannot get the target number of cluster. The input network may be incompatible with the specified number of clusters ...')
                break
            # print(len(test_partition))
            if len(test_partition) < 0.8 * maxn:
                if last_move == 'down':
                    maxres = np.sqrt(maxres * last_res)
                else:
                    maxres = maxres * 2.0
                last_move = 'up'
            elif len(test_partition) > 1.2 * maxn:
                if last_move == 'up':
                    maxres = np.sqrt(maxres * last_res)
                else:
                    maxres = maxres / 2.0
                last_move = 'down'
            else:
                next = False

    stack_res_range = []
    LOGGER.info('Lower bound of resolution parameter: {:.4f}'.format(minres))
    LOGGER.info('Upper bound of resolution parameter: {:.4f}'.format(maxres))
    stack_res_range.append((minres, maxres))

    _ = update_resolution_graph(resolution_graph, minres, density, neighbors)
    _ = update_resolution_graph(resolution_graph, maxres, density, neighbors)

    LOGGER.report('Resolution range initialized in %.2fs', '_resrange')

    # parallel: determine a sequence of resolutions
    all_resolutions = [minres, maxres]

    LOGGER.timeit('_sample')
    while stack_res_range:
        current_range = stack_res_range.pop(0)
        resname1, resname2 = '{:.4f}'.format(current_range[0]), '{:.4f}'.format(current_range[1])
        # LOGGER.debug('Current resolution range:{} {}'.format(resname1, resname2))

        if round(current_range[1] - current_range[0], 4) <= min_diff_resolution:
            # LOGGER.debug('Reaching the minimum difference between resolutions')
            continue
        if resolution_graph.nodes[resname1]['padded'] and resolution_graph.nodes[resname2]['padded']:
            continue

        # sample new resolutions and generate more partitions
        new_resolution = np.round(np.sqrt(current_range[1] * current_range[0]), 4)

        stack_res_range.append((current_range[0], new_resolution))
        stack_res_range.append((new_resolution, current_range[1]))

        all_resolutions.append(new_resolution)

        _ = update_resolution_graph(resolution_graph, new_resolution, density, neighbors)

        # cluG.add_clusters(resolution_graph, new_resolution)

    # run community detection for each resolution
    _arg_tuples = [(Gs, alg, res, sample, layer_weights) for res in all_resolutions]
    with mp.Pool(processes=numthreads) as pool:
        results = pool.starmap(run_alg, _arg_tuples)  # results contains "partition" class
    for i in range(len(all_resolutions)):
        nodename = '{:.4f}'.format(all_resolutions[i])
        resolution_graph.nodes[nodename]['matrix'] = results[i]
        cluG.add_clusters(resolution_graph, all_resolutions[i])

    # collapse related clusters
    LOGGER.report('Multiresolution Louvain clustering in %.2fs', '_sample')
    return cluG


def consensus(cluG, k=5, f=1.0, p=100):
    '''
    create a more parsimonious results from the cluster graph

    Parameters
    ----------
    cluG: hidef_finder.ClusterGraph
    k: int
        delete components with lower size than this threshold. The 'chi' parameter in the paper
    f: float
        take this fraction of clusters (ordered by degree in cluster graph)
    p: int
        nodes that do not participate in the majority of clusters in a component will be removed

    Returns
    ----------
    cluG_collapsed_w_len: a list of tuple (np.array, int)
        the array indicates the member of clusters; the integer represents the persistence of each

    See Also
    ----------
    collapse_cluster_graph

    '''
    from networkx.algorithms.community import k_clique_communities

    components = [c for c in nx.connected_components(cluG) if len(c) >= k]
    components = sorted(components, key=len, reverse=True)
    components_new = []
    # use k-clique percolation to recalculate components
    for component in components:
        component = list(component)
        clusters = [cluG.nodes[v]['data'].binary for v in component]
        mat = np.stack(clusters)
        matsp = sp.sparse.coo_matrix(mat, )
        matsp = matsp.tocsr()
        jacmat = jaccard_matrix(matsp, matsp)

        Gcli = nx.Graph()
        for i in range(len(jacmat[0])):
            na, nb = jacmat[0][i], jacmat[1][i]
            if na != nb:
                Gcli.add_edge(na, nb)

        clic_percolation = list(k_clique_communities(Gcli, k))  # this parameter better to stay
        for clic in clic_percolation:
            clic = list(clic)
            original_nodes = [component[c] for c in clic]
            components_new.append(set(original_nodes))

    components = components_new.copy()
    components = sorted(components, key=len, reverse=True)

    ntaken = int(f * len(components))
    components = components[:ntaken]  #

    cluG_collapsed = collapse_cluster_graph(cluG, components, p)
    len_components = [len(c) for c in components]

    cluG_collapsed_w_len = [(cluG_collapsed[i], len_components[i]) for i in range(len(cluG_collapsed))]
    cluG_collapsed_w_len = sorted(cluG_collapsed_w_len, key=lambda x: np.sum(x[0]),
                                  reverse=True)  # sort by cluster size

    return cluG_collapsed_w_len


def output_nodes(wv, names, out, extra_data=None, original_cluster_names=None):
    '''
    Write a .nodes file according to the weaver result. Four columns are: community names, sizes, member genes, and persistence

    Parameters
    ----------
    wv : weaver.Weaver

    names : list of string
        a list of ordered gene names
    out : string
        prefix of the output file
    extra_data: list of int
        a list of numbers to show on the last column of the output file
    original_cluster_names: list
        if not None, do not use the weaver renames, but use a list of specified names; should be equal to the number of clusters in "wv"

    Returns
    ----------
    '''
    wv_clusts = []
    if original_cluster_names != None:
        assert len(original_cluster_names) == len(wv._assignment)
    for v, vdata in wv.hier.nodes(data=True):
        if not isinstance(v, tuple):
            continue
        ind = vdata['index']
        if original_cluster_names != None:
            name = original_cluster_names[ind]
        else:
            name = 'Cluster{}-{}'.format(str(v[0]), str(v[1]))

        if extra_data != None:
            if isinstance(ind, int):
                persistence = extra_data[ind]
                wv_clusts.append([name, wv._assignment[ind], persistence])
            else:
                persistence = sum(
                    [extra_data[x] for x in ind])  # TODO: this tuple is a bit dumb. Should be inside the program
                wv_clusts.append([name, wv._assignment[ind[0]], persistence])
        else:
            if isinstance(ind, int):
                wv_clusts.append([name, wv._assignment[ind]])
            else:
                wv_clusts.append([name, wv._assignment[ind[0]]])
    wv_clusts = sorted(wv_clusts, key=lambda x: np.sum(x[1]), reverse=True)

    with open(out + '.nodes', 'w') as fh:

        for ci in range(len(wv_clusts)):
            cn = wv_clusts[ci][0]
            cc = wv_clusts[ci][1]
            if extra_data != None:
                cl = wv_clusts[ci][2]
                fh.write(cn + '\t' + str(np.sum(cc)) + '\t' + ' '.join(
                    sorted([names[x] for x in np.where(cc)[0]])) + '\t' + str(cl) + '\n')
            else:
                fh.write(cn + '\t' + str(np.sum(cc)) + '\t' + ' '.join(
                    sorted([names[x] for x in np.where(cc)[0]])) + '\n')
    return


def output_edges(wv, names, out, leaf=False, original_cluster_names=None):
    '''
    Output hierarchy in the DDOT format; wst column is parents and 2nd column is children
    Note this output is the 'forward' state.

    Parameters
    ----------
    wv : weaver.Weaver
    names : list of string
        a list of ordered gene names
    out : string
        prefix of the output file
    leaf: bool
        if True, then write genes into the result
    original_cluster_names: list
        if not None, do not use the weaver renames, but use a list of specified names; should be equal to the number of clusters in "wv"

    Returns
    ----------
    '''
    # note this output is the 'forward' state
    # right now do not support node names as tuples
    with open(out + '.edges', 'w') as fh:
        for e in wv.hier.edges():
            if original_cluster_names == None:
                parent = 'Cluster{}'.format(str(e[0][0]) + '-' + str(e[0][1]))
                if isinstance(e[1], tuple):
                    child = 'Cluster{}-{}'.format(str(e[1][0]), str(e[1][1]))
                    outstr = '{}\t{}\tdefault\n'.format(parent, child)
                    fh.write(outstr)
                elif leaf:
                    child = names[e[1]]
                    outstr = '{}\t{}\tgene\n'.format(parent, child)
                    fh.write(outstr)
            else:
                parent = original_cluster_names[wv.hier.nodes[e[0]]['index']]
                if isinstance(e[1], tuple):
                    child = original_cluster_names[wv.hier.nodes[e[1]]['index']]
                    outstr = '{}\t{}\tdefault\n'.format(parent, child)
                    fh.write(outstr)
                elif leaf:
                    child = names[e[1]]
                    outstr = '{}\t{}\tgene\n'.format(parent, child)
                    fh.write(outstr)


def output_gml(out):
    '''
    write a GML file for the hierarchy.

    Parameters
    ----------
    out: string
        prefix of the output file

    Returns
    ----------
    '''
    assert os.path.isfile(out + '.edges') and os.path.isfile(out + '.nodes'), 'Files do not exist'
    df_node = pd.read_csv(out + '.nodes', sep='\t', index_col=0, header=None)
    df_edge = pd.read_csv(out + '.edges', sep='\t', header=None)
    df_edge = df_edge.loc[df_edge[2] == 'default', :]
    if df_node.shape[1] == 3:
        df_node.columns = ['Size', 'MemberList', 'Stability']
    else:
        df_node.columns = ['Size', 'MemberList']
    df_node['LogSize'] = np.log2(df_node['Size'])

    G = nx.from_pandas_edgelist(df_edge, source=0, target=1, create_using=nx.DiGraph())
    for c in df_node.columns:
        dic = df_node[c].to_dict()
        nx.set_node_attributes(G, dic, c)
    nx.write_gml(G, out + '.gml')


def output_all(wv, names, out, persistence=None, iter=False, skipgml=False, t=0.75):
    if iter == False:
        output_nodes(wv, names, out, persistence)
    else:
        nnode_old, nedge_old = 0, 0
        nnode, nedge = len(wv.hier.nodes()), len(wv.hier.edges())
        i = 1
        output_nodes(wv, names, '_tmp.{}.'.format(i) + out, extra_data=persistence)
        while (nnode != nnode_old) and (nedge != nedge_old):
            g2ind = {names[k]: k for k in range(len(names))}
            if persistence != None:
                data = node2mat('_tmp.{}.'.format(i) + out + '.nodes', g2ind, has_persistence=True)
            else:
                data = node2mat('_tmp.{}.'.format(i) + out + '.nodes', g2ind)
            wv = weaver.Weaver()
            wv.weave(data['cluster'], boolean=True, merge=True, cutoff=t)
            nnode_old, nedge_old = nnode, nedge
            nnode, nedge = len(wv.hier.nodes()), len(wv.hier.edges())
            i += 1
            output_nodes(wv, names, '_tmp.{}.'.format(i) + out, extra_data=data['extra.data'])
        os.system('mv _tmp.{}.{}.nodes {}.nodes'.format(i, out, out))
        os.system('rm _tmp*.nodes')

    output_edges(wv, names, out)
    if skipgml is False:
        output_gml(out)

    return wv  # return the last step of weaver


if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--g', required=True, nargs='+', help='The input graph as a TSV file (no header). If there are '
                                                           'multiple input files, will run multiplex mode')
    par.add_argument('--minres', type=float, default=0.001, help='Minimum resolution parameter')  # cdaps not-expose
    par.add_argument('--maxres', type=float, default=25.0, help='Maximum resolution parameter. Increase to get more '
                                                                'smaller communities.')  # Maximum resolution parameter
    par.add_argument('--n', type=int, help='The target community number. Explore the maximum resolution parameter '
                                           'until the number of generated communities at this resolution is close '
                                           'enough to this value. Increase to get more smaller communities.')  #
    # Target community number
    par.add_argument('--d', type=float, default=0.1, help='Inversed density of sampling the resolutions. Decrease to '
                                                          'sample more resolutions and introduce more transient '
                                                          'communities (will increase running time)')  # cdaps
    # not-expose
    par.add_argument('--t', type=float, default=0.75, help='The tau parameter; the similarity/containment threshold. '
                                                           'A cutoff for creating the community ensemble graph and '
                                                           'the containment graph')  # cdaps not-expose
    par.add_argument('--k', type=int, default=5, help='The chi parameter; Persistence threshold. Increase to delete '
                                                      'unstable clusters and get fewer communities')  # Persistent
    # threshold.
    par.add_argument('--s', type=float, default=1.0, help='A subsample parameter')  # cdaps not-expose
    par.add_argument('--p', default=75, type=int, help='The p parameter; the consensus threshold collapsing community '
                                                       'graph and choose representative genes for each community '
                                                       'ensemble')  # Consensus threshold.
    par.add_argument('--o', required=True, help='output file in ddot format')
    par.add_argument('--alg', default='leiden', choices=['louvain', 'leiden'], help='accept louvain or leiden')
    par.add_argument('--iter', action='store_true', help='iterate weave function until fully converge')
    par.add_argument('--skipgml', action='store_true', help='If True, skips output of gml file')
    par.add_argument('--keepclug', action='store_true', help='If True, output of cluG file')
    par.add_argument('--numthreads', type=int, help='Number of child threads to spawn for clustering. '
                                                    'Value of multiprocessing.cpu_count() will be used '
                                                    'if NUMTHREADS is not set, is zero, or is negative')
    par.add_argument('--layer_weights', nargs='*', type=float, help='if multiple values, use multiplex mode; if not '
                                                                    'set but there are multiple input networks, '
                                                                    'the weight of each layer is 1')

    args = par.parse_args()

    # if multiple networks, need to harmonize the node order
    layer_weights = None
    if len(args.g) > 1:
        if (args.layer_weights is None) or (len(args.layer_weights) == 0):
            layer_weights = [1.0 for _ in args.g]
        else:
            layer_weights = args.layer_weights
        tmpfiles = []
        nodenames = []
        for G in args.g:
            df = pd.read_csv(G, sep='\t', header=None)
            nodenames.extend(df[0].astype(str).tolist() + df[1].astype(str).tolist())
        nodenames = sorted(set(nodenames))
        nodedict = {x: i for i, x in enumerate(nodenames)}
        max_i = len(nodenames) - 1
        for G in args.g:
            tmpf = '_tmp.' + os.path.basename(G)
            with open(G) as fh, open(tmpf, 'w') as ofh:
                for l in fh:
                    ll = l.strip().split()
                    li1, li2 = nodedict[ll[0]], nodedict[ll[1]]
                    ofh.write(str(li1) + '\t' + str(li2) + '\n')
                ofh.write(str(max_i) + '\t' + str(max_i) + '\n')  # this is to indicate the number of nodes the
                # network should have, even no edges are connected. This is a self edge and won't affect the result
            tmpfiles.append(tmpf)
        Gs = [ig.Graph.Read_Edgelist(G, directed=False) for G in tmpfiles]
    else:
        Gs = [ig.Graph.Read_Ncol(G, directed=False) for G in args.g]

    # G_component = list(G.components(mode='WEAK'))
    # if args.n != None:
    #     args.n = args.n + len(G_component) - 1

    # for now it requires G.vs is identical for all networks

    if args.numthreads is None or args.numthreads <= 0 or args.numthreads > mp.cpu_count():
        num_threads = mp.cpu_count()
    else:
        num_threads = args.numthreads
    # explore the resolution parameter given the number of clusters

    cluG = run(Gs,
               density=args.d,
               jaccard=args.t,
               sample=args.s,
               minres=args.minres,
               maxres=args.maxres,
               maxn=args.n,  # will soon go deprecated
               alg=args.alg,
               numthreads=num_threads,
               layer_weights=layer_weights
               )
    # # use weaver to organize them (due to the previous collapsed step, need to re-calculate containment index. This may be ok
    # components = sorted(nx.connected_components(cluG), key=len, reverse=True)
    if args.keepclug:
        filename = args.o + '.cluG'
        outfile = open(filename, 'wb')
        pickle.dump(cluG, outfile)
        outfile.close()

    LOGGER.timeit('_consensus')
    cluG_collapsed_w_len = consensus(cluG, args.k, 1.0, args.p)  # have sorted by cluster size inside this function
    cluG_collapsed = [x[0] for x in cluG_collapsed_w_len]
    len_component = [x[1] for x in cluG_collapsed_w_len]
    cluG_collapsed.insert(0, np.ones(len(cluG_collapsed[0]), ))
    len_component.insert(0, 0)
    LOGGER.report('Processing cluster graph in %.2fs', '_consensus')

    wv = weaver.Weaver()
    T = wv.weave(cluG_collapsed, boolean=True, levels=False,
                 merge=True, cutoff=args.t)  #

    if len(args.g) == 1:
        names = [Gs[0].vs[i]['name'] for i in range(len(Gs[0].vs))]
    else:
        names = nodenames

    output_all(wv, names, args.o, persistence=len_component, iter=args.iter, skipgml=args.skipgml)
