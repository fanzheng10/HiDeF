import louvain
import networkx as nx
import igraph as ig
import argparse
import numpy as np
import scipy as sp
import time
from weaver import *


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
        # jaccard index
        arr1, arr2 = self.binary, cluster2.binary
        both = np.dot(arr1, arr2)
        either = len(arr1) - np.dot(1-arr1, 1-arr2)
        return 1.0 * both / either

class ClusterGraph(nx.Graph): # inherit networkx digraph

    def add_clusters(self, resolution_graph, new_resolution):
        resname_new = '{:.4f}'.format(new_resolution)
        new_clusters = []
        new_mat = resolution_graph.nodes[resname_new]['matrix']

        for i in range(new_mat.shape[0]): # c is a list of node indices
            clu = Cluster(new_mat[i, :], self.graph['num_leaves'], new_resolution)
            # cluG.add_cluster(clu)
            new_clusters.append(clu)

        # print('add {:d} clusters'.format(len(new_clusters)))

        # compare c with all existing clusters in clusterGraph (maybe optimized in future)
        if len(self.nodes()) == 0:
            newnode = np.arange(0, len(new_clusters))
        else:
            newnode = np.arange(max(self.nodes()) +1, max(self.nodes()) +1 + len(new_clusters))
        resolution_graph.nodes[resname_new]['node_indices'] = newnode


        # newedges = []
        # for nci in range(len(new_clusters)):
        #     nc = new_clusters[nci]
        #     ni = newnode[nci]
        #     for i, c in self.nodes.items():
        #         resname_c = c['data'].resolution_parameter
        #         if resname_c in resolution_graph[resname_new]:
        #             if 1.0 * min(nc.size, c['data'].size)/max(nc.size, c['data'].size) < self.graph['sim_threshold'] :
        #                 continue
        #             similarity = nc.calculate_similarity(c['data'])
        #             if similarity > self.graph['sim_threshold']:
        #                 newedges.append((ni, i, similarity))

        # compare against all other resolutions within range
        newedges = []
        for r in resolution_graph.nodes():
            if r == resname_new: # itself
                continue
            if r in resolution_graph[resname_new]: # resolution in close proximity
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

    def drop_cluster(self, nodes):
        # print('remove {:d} lonely clusters'.format(len(nodes)))
        self.remove_nodes_from(nodes)

    def remove_clusters(self, k, coherence=0.5):
        # find a k-core of the cluster graph
        nodes_to_remove = []
        core_numbers = nx.core_number(self)
        for i, v in core_numbers.items():
            clust = self.nodes[i]['data']
            if clust.padded and v < coherence*k:
                nodes_to_remove.append(i)
        self.drop_cluster(nodes_to_remove)

    def update_padding(self, newly_padded_resolution):
        for i, c in self.nodes.items():
            clust = self.nodes[i]['data']
            if clust.resolution_parameter in newly_padded_resolution:
                clust.padded = True

def jaccard_matrix(matA, matB, threshold=0.75, prefilter = False): # assume matA, matB are sorted
    # prefiltered version has bugs and doesn't seem to to
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


def run_alg(G, gamma=1.0):
    '''
    run community detection algorithm with resolution parameter. Right now only use RB in Louvain
    :param G: an igraph graph
    :param gamma: resolution parameter
    :return: 
    '''
    partition_type = louvain.RBConfigurationVertexPartition
    partition = louvain.find_partition(G, partition_type, resolution_parameter=gamma)
    # partition = sorted(partition, key=len, reverse=True)
    return partition

def network_perturb(G, sample=0.8):
    G1 = G.copy()
    edges_to_remove = [e.index for e in G1.es if np.random.rand() > sample]
    G1.delete_edges(edges_to_remove)
    return G1

def partition_to_membership_matrix(partition, minsize=4):
    clusters = sorted([p for p in partition if len(p) >=minsize], key=len, reverse=True)
    row, col = [], []
    for i in range(len(clusters)):
        row.extend([i for _ in clusters[i]])
        col.extend([x for x in clusters[i]])
    row = np.array(row)
    col = np.array(col)
    data = np.ones_like(row, dtype=int)
    C = sp.sparse.coo_matrix((data, (row, col)), shape=(len(clusters), partition.n))
    C = C.tocsr()
    return C


def update_resolution_graph(G, new_resolution, partition, value, neighborhood_size, neighbor_density_threshold):
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

# def ensure_monotonicity(bisect_values, new_res):
#     # First check if this partition improves on any other partition
#     for res, bisect_part in bisect_values.iteritems():
#         if bisect_values[new_res].partition.quality(res) > bisect_part.partition.quality(res):
#             bisect_values[res] = bisect_values[new_res]
#     # Then check what is best partition for the new_res
#     current_quality = bisect_values[new_res].partition.quality(new_res)
#     best_res = new_res
#     for res, bisect_part in bisect_values.iteritems():
#         if bisect_part.partition.quality(new_res) > current_quality:
#             best_res = new_res
#     bisect_values[new_res] = bisect_values[best_res]

def collapse_cluster_graph(cluG, components, threshold=100):
    '''
    take the cluster graph and collapse each component based on some consensus metric
    :param components: a list of list, each element of the inner list is a binary membership array
    :param threshold: t; remove nodes if they did not appear in more than t percent of clusters in one component 
    :return: collapsed_clusters: a list of arrays, of which the length is equal to that of components
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
        maxn=None,
        bisect=False):
    # other default parameters
    min_diff_bisect_value = 1
    min_diff_resolution = 0.0001

    # G = ig.Graph.Read_Ncol(G)
    G.simplify(multiple=False) # remove self loop but keep weight
    cluG = ClusterGraph()
    cluG.graph['sim_threshold'] = jaccard
    cluG.graph['num_leaves'] = len(G.vs)

    resolution_graph = nx.Graph()

    # perform two initial louvain
    minres_partition = run_alg(G, minres)

    if maxn != None:
        next= True
        last_move = ''
        last_res = maxres
        while next==True:
            test_partition = run_alg(G, maxres)
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
        maxres_partition = run_alg(G, maxres)
    stack_res_range = []
    print('Lower bound of resolution parameter: {}'.format(minres))
    print('Upper bound of resolution parameter: {:.4f}'.format(maxres))
    stack_res_range.append((minres, maxres))

    update_resolution_graph(resolution_graph, minres, minres_partition,
                            minres_partition.total_weight_in_all_comms(), density, neighbors)
    update_resolution_graph(resolution_graph, maxres, maxres_partition,
                            maxres_partition.total_weight_in_all_comms(), density, neighbors)
    cluG.add_clusters(resolution_graph, minres)
    cluG.add_clusters(resolution_graph, maxres)

    # start = time.time()
    while stack_res_range:
        current_range = stack_res_range.pop(0)
        resname1, resname2 = '{:.4f}'.format(current_range[0]), '{:.4f}'.format(current_range[1])
        # print('Current resolution range:', resname1, resname2)

        if round(current_range[1] - current_range[0], 4) <= min_diff_resolution:
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
        # print('New resolution:', resname_new)

        stack_res_range.append((current_range[0], new_resolution))
        stack_res_range.append((new_resolution, current_range[1]))
        if sample<1:
            G1 = network_perturb(G, sample)
            new_partition = run_alg(G1, new_resolution)
        else:
            new_partition = run_alg(G, new_resolution)
        _ = update_resolution_graph(resolution_graph, new_resolution, new_partition, new_partition.total_weight_in_all_comms(), density, neighbors)
        # if len(newly_padded_resolution) > 0:
        #     print(newly_padded_resolution)
        cluG.add_clusters(resolution_graph, new_resolution)

        # print('time elapsed: {:.3f}'.format(time.time() - start))
        # cluG.update_padding(newly_padded_resolution) # think about here. The third parameter needs to be larger than k

        # print('time elapsed: {:.3f}'.format(time.time() - start))
        # cluG.remove_clusters(neighbors)
         # in
        # print('time elapsed: {:.3f}'.format(time.time() - start))

    # collapse related clusters
    return cluG

def consensus(cluG, k,  f, ct):
    nodes_to_remove = []
    core_numbers = nx.core_number(cluG)
    for i, v in core_numbers.items():
        if v < 0.5 * k:
            nodes_to_remove.append(i)
    cluG2 = cluG.copy()
    cluG2.remove_nodes_from(nodes_to_remove)

    components = [c for c in nx.connected_components(cluG2)]
    components = sorted(components, key=len, reverse=True)
    ntaken = f * len(components)
    components = components[:ntaken]  #
    # csize = len(components[0]) * f
    # components = [c for c in components if len(c) >= csize]
    # TODO: whether it should be smarter than that?

    cluG_collapsed = collapse_cluster_graph(cluG2, components, ct)
    cluG_collapsed = sorted(cluG_collapsed, key=lambda x: np.sum(x), reverse=True)  # sort by cluster size
    return cluG_collapsed

def output_cluster(weaver, G, out, sort='int'):
    internals = lambda T: (node for node in T if isinstance(node, tuple))
    weaver_clusts = []
    for node in internals(weaver.hier.copy()):
        weaver_clusts.append((node, weaver.node_cluster(node)))
    weaver_clusts = sorted(weaver_clusts, key=lambda x: np.sum(x[1]), reverse=True)

    # first a cluster memberhsip file, this is for Seurat visualization
    if sort=='int':
        ind_order = sorted(np.arange(len(G.vs)), key=lambda x: int(G.vs[x]['name']))
    else:
        ind_order = sorted(np.arange(len(G.vs)), key=lambda x: G.vs[x]['name'])
    for ci in range(len(weaver_clusts)):
        weaver_clusts[ci][1] = np.array(weaver_clusts[ci][1])[ind_order]

    with open(out + '.membership', 'w') as fh:
        for ci in range(len(weaver_clusts)):
            cn = '_'.join(map(str, weaver_clusts[ci][0]))
            cc = weaver_clusts[ci][1]
            fh.write('Cluster_{}\t'.format(cn) + str(np.sum(cc)) + '\t' + ','.join(
                (np.where(cc)[0] + 1).astype(str)) + '\n')

def output_ddot(weaver, G, out, leaf = False):
    with open(out+ '.ddot', 'w') as fh:
        for e in weaver.hier.edges():
            if isinstance(e[1], tuple):
                outstr = 'Cluster_{}\tCluster_{}\tInternal\n'.format('_'.join(map(str, e[0])), '_'.join(map(str, e[1])))
                fh.write(outstr)
            elif leaf:
                outstr = 'Cluster_{}\t{}\tLeaf\n'.format('_'.join(map(str, e[1])), G.vs[e[1]]['name'])
                fh.write(outstr)

def output_cytoscape(weaver, G, out):

    dict_node_size = {n :weaver.node_size(n) for n in weaver.hier.nodes() if isinstance(n, tuple)}
    nx.set_node_attributes(weaver.hier, dict_node_size, 'size')

    # remove all leaf nodes
    Gout = weaver.hier.copy()
    nodes_to_remove = [n for n in Gout.nodes() if not isinstance(n, tuple)]
    Gout.remove_nodes_from(nodes_to_remove)

    # relabel nodes (tuple not compatible)
    mapping = {n: str(n[0]) +'_'+ str(n[1]) for n in Gout.nodes()}
    Gout = nx.relabel_nodes(Gout, mapping, copy=False)

    nx.write_gml(Gout, out +".gml")

    # TODO: upload to NDEx

if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--g', required=True, help='a tab separated file for the input graph')
    par.add_argument('--f', required=True, type=float, help='a parameter controlling the complexity of the hierarchy. From 0 to 1')
    par.add_argument('--n', default=50, type=int, help= 'an intuitive  parameter for the upper limit of resolution parameter')
    par.add_argument('--t', type=float, default=0.1, help='a parameter used in removal of lonely clusters; a lonely cluster (determined by --k) will be removed if some x samples have been sampled within +/- t of the current resolution') # since x is relative to t, doesn't have to set another parameter for x #
    par.add_argument('--k', type=int, default = 10, help='a parameter to calculate how much lonely nodes to remove (i.e. retain a k-core);')
    par.add_argument('--j', type=float, default=0.75, help='a jaccard index cutoff')
    # min and max resolution
    par.add_argument('--minres', type=float, default=0.0001)
    par.add_argument('--maxres', type=float, default=100.0)
    par.add_argument('--s', type=float, default=1.0, help='a subsample parameter')
    par.add_argument('--ct', default=100, type=int, help='threshold in collapsing cluster')
    par.add_argument('--o', required=True, help='output file in ddot format')
    par.add_argument('--sort', default='int', choices=['int', 'str'])
    args = par.parse_args()

    G = ig.Graph.Read_Ncol(args.g) # redundant

    # explore the resolution parameter given the number of clusters


    cluG = run(G,
               density=args.t,
               neighbors=args.k,
               jaccard=args.j,
               sample=args.s,
               minres=args.minres,
               maxres=args.maxres,
               maxn=args.n
               )
    # # use weaver to organize them (due to the previous collapsed step, need to re-calculate containment index. This may be ok
    # components = sorted(nx.connected_components(cluG), key=len, reverse=True)
    cluG_collapsed = consensus(cluG, args.k, args.f, args.ct)

    weaver = Weaver(cluG_collapsed, boolean=True, assume_levels=True)
    T = weaver.weave() #

    output_cluster(weaver, G, args.o, sort=args.sort)
    output_ddot(weaver, G, args.o)
    output_cytoscape(weaver, G, args.o)


