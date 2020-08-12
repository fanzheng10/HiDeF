#! /usr/bin/env python

import louvain
import leidenalg

import networkx as nx
import igraph as ig
import argparse
import time
import pickle
import numpy as np
import pandas as pd
import scipy as sp
from networkx.algorithms.community import k_clique_communities
from hidef import weaver
from hidef import LOGGER


class Cluster(object):
    __slots__ = ['size',
                 'members',
                 'binary',
                 'padded',
                 'resolution_parameter']

    def __init__(self, binary, length, gamma):
        '''initialize
        member: a list of member index (start from 0)
        size: number of cluster member
        length:  
        gamma: resolution parameter
         '''
        self.binary = np.squeeze(np.asarray(binary.todense()))
        self.members = np.where(self.binary)[0]
        self.size = len(self.members)
        self.resolution_parameter = '{:.4f}'.format(gamma)
        self.padded = False
        # self.index=None


    def calculate_similarity(self, cluster2):
        '''
        calculate Jaccard similarity.
        :param cluster2: another Cluster object
        :return: 
        '''
        # jaccard index
        arr1, arr2 = self.binary, cluster2.binary
        both = np.dot(arr1, arr2)
        either = len(arr1) - np.dot(1-arr1, 1-arr2)
        return 1.0 * both / either

class ClusterGraph(nx.Graph): # inherit networkx digraph
    '''
    Extending nx.Graph class, each node is a Cluster object
    '''

    def add_clusters(self, resolution_graph, new_resolution):
        '''
        Add new clusters to cluster graph once a new resolution is finished by the CD algorithm
        :param resolution_graph: another nx.Graph object 
        :param new_resolution: the resolution just visited by the CD algorithm
        :return: 
        '''
        resname_new = '{:.4f}'.format(new_resolution)
        new_clusters = []
        new_mat = resolution_graph.nodes[resname_new]['matrix']

        for i in range(new_mat.shape[0]): # c is a list of node indices
            clu = Cluster(new_mat[i, :], self.graph['num_leaves'], new_resolution)
            # cluG.add_cluster(clu)
            new_clusters.append(clu)


        # compare c with all existing clusters in clusterGraph (maybe optimized in future)
        if len(self.nodes()) == 0:
            newnode = np.arange(0, len(new_clusters))
        else:
            newnode = np.arange(max(self.nodes()) +1, max(self.nodes()) +1 + len(new_clusters))
        resolution_graph.nodes[resname_new]['node_indices'] = newnode

        # compare against all other resolutions within range
        newedges = []
        for r in resolution_graph.neighbors(resname_new):
            if r == resname_new: # itself
                continue
            r_node_ids = resolution_graph.nodes[r]['node_indices']
            rmat = resolution_graph.nodes[r]['matrix']
            id_new, id_r= jaccard_matrix(new_mat, rmat, self.graph['sim_threshold'])
            if len(id_new) > 0:
                id_new =  newnode[id_new]
                id_r = r_node_ids[id_r]
                for i in range(len(id_new)):
                    newedges.append((id_new[i], id_r[i]))

        # print('add {:d} new edges'.format(len(newedges)))
        # for ni, i, s in newedges:
        self.add_edges_from(newedges)

        for nci in range(len(new_clusters)):
            nc = new_clusters[nci]
            ni = newnode[nci]
            if not ni in self.nodes:
                self.add_node(ni)
            self.nodes[ni]['data'] = nc # is it a pointer?
            # self.nodes[ni]['data'].index = ni

    def remove_clusters(self, k, coherence=0.5):
        '''
        deprecated
        :param k: 
        :param coherence: 
        :return: 
        '''
        # find a k-core of the cluster graph
        nodes_to_remove = []
        core_numbers = nx.core_number(self)
        for i, v in core_numbers.items():
            clust = self.nodes[i]['data']
            if clust.padded and v < coherence*k:
                nodes_to_remove.append(i)
        self.remove_nodes_from(nodes_to_remove)


def jaccard_matrix(matA, matB, threshold=0.75, prefilter = False): # assume matA, matB are sorted
    '''
    calculate jaccard matrix between all pairs between two sets of clusters
    :param matA: scipy.sparse.csr_matrix, axis 0 for clusters, axis 1 for nodes in network
    :param matB: similar to matA; cluster set under a different resolution parameter
    :param threshold: a Jaccard similarity cutoff
    :param prefilter: pre-filter the pairs to compare with a lower bound (didn't seem to speed up much)
    :return: two sets of indices; the cluster pairs implied by those indices satisfied threshold
    '''
    if prefilter: # does not have much speed advantage
        sizeA = np.ravel(np.asarray(np.sum(matA, axis=1)))
        sizeB = np.ravel(np.asarray(np.sum(matB, axis=1)))
        idA, idB = [], []
        for i in range(len(sizeA)):
            for j in range(len(sizeB)):
                if 1.0 * min(sizeA[i], sizeB[j])/max(sizeA[i], sizeB[j]) < threshold:
                    continue
                else:
                    idA.append(i)
                    idB.append(j)
        if len(idA) > 0:
            idA = np.array(idA)
            idB = np.array(idB)
            # print(idA, idB)
            matAs, matBs = matA[idA, :], matB[idB, :]
            both = np.sum(matAs.multiply(matBs), axis=1).ravel()
            either = matAs.getnnz(axis=1) + matBs.getnnz(axis=1) - both
            # print(both.shape, matAs.getnnz(axis=1).shape,matBs.getnnz(axis=1).shape, np.min(either))
            jac = 1.0*both/either
            small_index = np.where(jac > threshold)
            # print(small_index)
            real_index = (idA[small_index[1]], idB[small_index[1]])
            return real_index
        else:
            return ([], [])
    else:
        both = matA.dot(matB.T)

        either = (np.tile(matA.getnnz(axis=1), (matB.shape[0],1)) + matB.getnnz(axis=1)[:, np.newaxis]).T -both
        jac = 1.0*both/either
        index = np.where(jac > threshold)
        return index


def run_alg(G, alg, gamma=1.0):
    '''
    run community detection algorithm with resolution parameter. Right now only use RB in Louvain
    :param G: an igraph graph
    :param gamma: resolution parameter
    :return: 
    '''
    if alg =='louvain':
        partition_type = louvain.RBConfigurationVertexPartition
        partition = louvain.find_partition(G, partition_type, resolution_parameter=gamma)
    elif alg == 'leiden':
        partition_type = leidenalg.RBConfigurationVertexPartition
        partition = leidenalg.find_partition(G, partition_type, resolution_parameter=gamma)
    # partition = sorted(partition, key=len, reverse=True)
    return partition

def network_perturb(G, sample=0.8):
    '''
    perturb the network by randomly deleting some edges
    :param G: input network
    :param sample: the fraction of edges to retain
    :return: the perturbed graph
    '''
    G1 = G.copy()
    edges_to_remove = [e.index for e in G1.es if np.random.rand() > sample]
    G1.delete_edges(edges_to_remove)
    return G1

def partition_to_membership_matrix(partition, minsize=4):
    '''
    
    :param partition: class partition in the louvain-igraph package
    :param minsize: minimum size of clusters; smaller clusters will be deleted afterwards 
    :return: 
    '''
    clusters = sorted([p for p in partition if len(p) >=minsize], key=len, reverse=True)
    row, col = [], []
    for i in range(len(clusters)):
        row.extend([i for _ in clusters[i]])
        col.extend([x for x in clusters[i]])
    row = np.array(row)
    col = np.array(col)
    data = np.ones_like(row, dtype=int)
    C = sp.sparse.coo_matrix((data, (row, col)), shape=(len(clusters), partition.n)) # TODO: make it not dependent on partition.n
    C = C.tocsr()
    return C


def update_resolution_graph(G, new_resolution, partition, value, neighborhood_size, neighbor_density_threshold):
    '''
    Update the "resolution graph", which connect resolutions that are close enough
    :param G: nx.Graph; the "resolution graph"
    :param new_resolution: the resolution just visited by the CD algorithm 
    :param partition: partition generated by 
    :param value: deprecated
    :param neighborhood_size: if two resolutions (log-scale) differs smaller than this value, they are called 'neighbors'
    :param neighbor_density_threshold: if a resolution has neighbors more than this number, it is called "padded". No more sampling will happen between two padded resolutions
    :return: 
    '''
    nodename = '{:.4f}'.format(new_resolution)
    membership = partition_to_membership_matrix(partition)
    G.add_node(nodename, resolution = new_resolution,
               matrix=membership,
               padded=False, value=value)
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

def collapse_cluster_graph(cluG, components, threshold=100):
    '''
    take the cluster graph and collapse each component based on some consensus metric
    :param cluG: the ClusterGraph object
    :param components: a list of list, each element of the inner list is a binary membership array
    :param threshold: t; remove nodes if they did not appear in more than t percent of clusters in one component 
    '''
    collapsed_clusters = []
    for component in components:
        clusters = [cluG.nodes[v]['data'].binary for v in component]
        mat = np.stack(clusters)
        participate_index = np.mean(mat, axis=0)
        threshold_met = participate_index *100 > threshold
        threshold_met = threshold_met.astype(int)
        collapsed_clusters.append(threshold_met)
    return collapsed_clusters

def run(G,
        density=0.1,
        neighbors=10,
        jaccard=0.75,
        sample=0.9,
        minres=0.01,
        maxres=10,
        alg='louvain',
        maxn=None,
        bisect=False):
    # other default parameters
    '''
    Main function to run the Finder program
    :param G: input network
    :param density: inversed density of sampling resolution parameter. Use a smaller value to increase sample density (will increase running time)
    :param neighbors: also affect sampling density; a larger value may have additional benefits of stabilizing clustering results
    :param jaccard: a cutoff to call two clusters similar
    :param sample: parameter to perturb input network in each run by deleting edges; lower values delete more
    :param minres: minimum resolution parameter
    :param maxres: maximum resolution parameter
    :param maxn: will explore resolution parameter until cluster number is similar to this number; will override 'maxres'
    :param bisect: if set to True, if solutions between two resolutions look similar, halt sampling in between. Could reduce stability a little
    :return: 
    '''
    min_diff_bisect_value = 1
    min_diff_resolution = 0.001

    # G = ig.Graph.Read_Ncol(G)
    G.simplify(multiple=False) # remove self loop but keep weight
    cluG = ClusterGraph()
    cluG.graph['sim_threshold'] = jaccard
    cluG.graph['num_leaves'] = len(G.vs)

    resolution_graph = nx.Graph()

    # perform two initial louvain
    minres_partition = run_alg(G, alg, minres)

    LOGGER.timeit('_resrange')
    LOGGER.info('Finding maximum resolution...')
    if maxn != None:
        next= True
        last_move = ''
        last_res = maxres
        maximum_while_loop = 20
        n_loop = 0
        while next==True:
            test_partition = run_alg(G, alg, maxres)
            n_loop += 1
            if n_loop > maximum_while_loop:
                LOGGER.warning(
                    'Reach upper limit of initial resolution searching, cannot get the target number of cluster. The input network may be incompatible with the specified number of clusters ...')
                break
            # print(len(test_partition))
            if len(test_partition) < 0.8*maxn:
                if last_move == 'down':
                    maxres = np.sqrt(maxres*last_res)
                else:
                    maxres = maxres*2.0
                last_move = 'up'
            elif len(test_partition) > 1.2*maxn:
                if last_move == 'up':
                    maxres = np.sqrt(maxres*last_res)
                else:
                    maxres = maxres/2.0
                last_move = 'down'
            else:
                next=False
        maxres_partition = test_partition
    else:
        maxres_partition = run_alg(G, alg, maxres)
    stack_res_range = []
    LOGGER.info('Lower bound of resolution parameter: {:.4f}; with {:d} clusters'.format(minres, len(minres_partition)))
    LOGGER.info('Upper bound of resolution parameter: {:.4f}; with {:d} clusters'.format(maxres, len(maxres_partition)))
    stack_res_range.append((minres, maxres))

    # TODO: resolution graph won't be needed. will be able to perform all-to-all comparison
    update_resolution_graph(resolution_graph, minres, minres_partition,
                            minres_partition.total_weight_in_all_comms(), density, neighbors)
    update_resolution_graph(resolution_graph, maxres, maxres_partition,
                            maxres_partition.total_weight_in_all_comms(), density, neighbors)
    cluG.add_clusters(resolution_graph, minres)
    cluG.add_clusters(resolution_graph, maxres)
    LOGGER.report('Resolution range initialized in %.2fs', '_resrange')


    LOGGER.timeit('_sample')
    while stack_res_range:
        current_range = stack_res_range.pop(0)
        resname1, resname2 = '{:.4f}'.format(current_range[0]), '{:.4f}'.format(current_range[1])
        # LOGGER.debug('Current resolution range:{} {}'.format(resname1, resname2))

        if round(current_range[1] - current_range[0], 4) <= min_diff_resolution:
            # LOGGER.debug('Reaching the minimum difference between resolutions')
            continue
        if bisect:
            if resolution_graph.nodes[resname1]['value'] - resolution_graph.nodes[resname2]['value'] < min_diff_bisect_value:
                # didn't do ensure monotonicity as Van Traag. not sure can that be a problem?
                continue
        if resolution_graph.nodes[resname1]['padded'] and resolution_graph.nodes[resname2]['padded']:
            continue

        # sample new resolutions and generate more partitions
        new_resolution = np.round(np.sqrt(current_range[1] * current_range[0]), 4)
        resname_new = '{:.4f}'.format(new_resolution)

        stack_res_range.append((current_range[0], new_resolution))
        stack_res_range.append((new_resolution, current_range[1]))
        if sample<1:
            G1 = network_perturb(G, sample)
            new_partition = run_alg(G1, alg, new_resolution)
        else:
            new_partition = run_alg(G, alg, new_resolution)

        LOGGER.info('Resolution:' + resname_new + '; find {} clusters'.format(len(new_partition)))

        _ = update_resolution_graph(resolution_graph, new_resolution, new_partition, new_partition.total_weight_in_all_comms(), density, neighbors)

        cluG.add_clusters(resolution_graph, new_resolution)

    # collapse related clusters
    LOGGER.report('Multiresolution Louvain clustering in %.2fs', '_sample')
    return cluG

def consensus(cluG, k=5,  f=1.0, ct=100):
    '''
    create a more parsimonious results from the cluster graph
    :param cluG: the cluster graph
    :param k: delete clusters with lower degree
    :param f: take this fraction of clusters (ordered by degree in cluster graph)
    :param ct: nodes that do not participate in the majority of clusters in a component will be removed
    :return: 
    '''

    components = [c for c in nx.connected_components(cluG) if len(c)>= k]
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


    cluG_collapsed = collapse_cluster_graph(cluG, components, ct)
    len_components = [len(c) for c in components]

    cluG_collapsed_w_len = [(cluG_collapsed[i], len_components[i]) for i in range(len(cluG_collapsed))]
    cluG_collapsed_w_len = sorted(cluG_collapsed_w_len, key=lambda x: np.sum(x[0]), reverse=True)  # sort by cluster size

    return cluG_collapsed_w_len

def output_nodes(weaver, G, out, len_component):
    # internals = lambda T: (node for node in T if isinstance(node, tuple))
    weaver_clusts = []
    for v, vdata in weaver.hier.nodes(data=True):
        if not isinstance(v, tuple):
            continue
        ind = vdata['index']
        name = 'Cluster{}-{}'.format(str(v[0]), str(v[1]))
        weaver_clusts.append([name, weaver._assignment[ind], len_component[ind]])
    weaver_clusts = sorted(weaver_clusts, key=lambda x: np.sum(x[1]), reverse=True)
    # TODO: in this output, the gene assignment has not been propagated. so it is possible that children genes are not in parent gene
    with open(out + '.nodes', 'w') as fh:
        # write parameter setting first
        # fh.write('# -f {}\n'.format(args.f))
        # fh.write('# -n {}\n'.format(args.n))
        # fh.write('# -t {}\n'.format(args.t))
        # fh.write('# -k {}\n'.format(args.k))
        # fh.write('# -j {}\n'.format(args.j))
        # fh.write('# -s {}\n'.format(args.s))
        # fh.write('# -ct {}\n'.format(args.ct))
        # fh.write('# -minres {}\n'.format(args.minres))
        # fh.write('# -maxres {}\n'.format(args.maxres))

        for ci in range(len(weaver_clusts)):
            cn = weaver_clusts[ci][0]
            cc = weaver_clusts[ci][1]
            cl = weaver_clusts[ci][2]
            fh.write(cn + '\t' + str(np.sum(cc)) + '\t' +
                     ' '.join(sorted([G.vs[x]['name'] for x in np.where(cc)[0] ])) + '\t' + str(cl) + '\n')
    return

def output_edges(weaver, G, out, leaf = False):
    '''
    
    :param weaver: 
    :param G: 
    :param out: 
    :param leaf: 
    :return: 
    '''
    # note this output is the 'forward' state
    # right now do not support node names as tuples
    with open(out+ '.edges', 'w') as fh:
        for e in weaver.hier.edges():
            parent = 'Cluster{}'.format(str(e[0][0]) + '-' + str(e[0][1]))
            if isinstance(e[1], tuple):
                child = 'Cluster{}-{}'.format(str(e[1][0]), str(e[1][1]))
                outstr = '{}\t{}\tdefault\n'.format(parent, child)
                fh.write(outstr)
            else:
                child = G.vs[e[1]]['name']
                if leaf:
                    outstr = '{}\t{}\tgene\n'.format(parent, child)
                    fh.write(outstr)


def output_gml(out):

    df_node = pd.read_csv(out + '.nodes', sep='\t', index_col=0, header=None)
    df_edge = pd.read_csv(out + '.edges', sep='\t', header=None)
    df_node.columns = ['Size', 'MemberList', 'Stability']
    df_node['LogSize'] = np.log2(df_node['Size'])

    G = nx.from_pandas_edgelist(df_edge, source=0, target=1, create_using=nx.DiGraph())
    for c in df_node.columns:
        dic = df_node[c].to_dict()
        nx.set_node_attributes(G, dic, c)
    nx.write_gml(G, out +'.gml')



if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--g', required=True, help='A tab separated file for the input graph')
    par.add_argument('--minres', type=float, default=0.001, help='Minimum resolution parameter') # cdaps not-expose
    par.add_argument('--maxres', type=float, default=50.0, help='Maximum resolution parameter. Increase to get more smaller communities.') # Maximum resolution parameter
    par.add_argument('--n', type=int, help= 'Target community number. Explore the maximum resolution parameter until the number of generated communities at this resolution is close enough to this value. Increase to get more smaller communities.') # Target community number
    par.add_argument('--t', type=float, default=0.1, help='Sampling density. Inversed density of sampling the resolution parameter. Decrease to introduce more transient communities (will increase running time)') # cdaps not-expose
    par.add_argument('--j', type=float, default=0.75, help='Similarity/containment threshold. A cutoff for creating the community ensemble graph and the containment graph') # cdaps not-expose
    par.add_argument('--k', type=int, default = 5, help='Persistence threshold. Increase to delete unstable clusters, and get fewer communities') # Persistent threshold.
    par.add_argument('--s', type=float, default=1.0, help='A subsample parameter') # cdaps not-expose
    par.add_argument('--ct', default=75, type=int, help='Consensus threshold. Threshold of collapsing community graph and choose genes for each community.') # Consensus threshold.
    par.add_argument('--o', required=True, help='output file in ddot format')
    par.add_argument('--alg', default='louvain', choices=['louvain', 'leiden'], help='add the option to use leiden algorithm')
    par.add_argument('--skipclug', action='store_true', help='If set, skips output of cluG file')
    par.add_argument('--skipgml', action='store_true', help='If set, skips output of gml file')

    args = par.parse_args()

    G = ig.Graph.Read_Ncol(args.g)
    G_component  = list(G.components(mode='WEAK'))
    if args.n != None:
        args.n = args.n + len(G_component) - 1

    # explore the resolution parameter given the number of clusters

    cluG = run(G,
               density=args.t,
               jaccard=args.j,
               sample=args.s,
               minres=args.minres,
               maxres=args.maxres,
               maxn=args.n,
               alg=args.alg
               )
    # # use weaver to organize them (due to the previous collapsed step, need to re-calculate containment index. This may be ok
    # components = sorted(nx.connected_components(cluG), key=len, reverse=True)
    if args.skipclug is False:
        filename = args.o + '.cluG'
        outfile = open(filename, 'wb')
        pickle.dump(cluG, outfile)
        outfile.close()

    LOGGER.timeit('_consensus')
    cluG_collapsed_w_len = consensus(cluG, args.k, 1.0, args.ct) # have sorted by cluster size inside this function
    cluG_collapsed = [x[0] for x in cluG_collapsed_w_len]
    len_component = [x[1] for x in cluG_collapsed_w_len]
    cluG_collapsed.insert(0, np.ones(len(cluG_collapsed[0]), ))
    len_component.insert(0, 0)
    LOGGER.report('Processing cluster graph in %.2fs', '_consensus')

    weaver = weaver.Weaver()
    T = weaver.weave(cluG_collapsed, boolean=True, assume_levels=False,
                     merge=True, cutoff=args.j) #

    output_nodes(weaver, G, args.o, len_component)
    output_edges(weaver, G, args.o)
    if args.skipgml is False:
        output_gml(args.o)