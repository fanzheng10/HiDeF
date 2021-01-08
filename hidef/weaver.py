import numpy as np
import scipy as sp
import networkx as nx
from collections import Counter, defaultdict
import itertools
from itertools import product as iproduct
from sys import getrecursionlimit, setrecursionlimit

from hidef.utils import *
from hidef import LOGGER

istuple = lambda n: isinstance(n, tuple)
isdummy = lambda n: None in n
internals = lambda T: (node for node in T if istuple(node))
RECURSION_MAX_DEPTH = int(10e6)

class Weaver(object):
    """Class for constructing a hierarchical representation of a graph
    based on (a list of) input partitions. """

    __slots__ = ['_assignment', '_terminals', 'assume_levels', 'hier', '_levels', '_labels',
                 '_full', '_secondary']

    def __init__(self):
        self.hier = None
        self._full = None
        self._secondary = None
        self._labels = None
        self._levels = None
        self.assume_levels = False
        self._terminals = None
        self._assignment = None

    def number_of_terminals(self):
        if self._assignment is None:
            return 0
        return len(self._assignment[0])

    n_terminals = property(number_of_terminals, 'the number of terminal nodes')    

    def set_terminals(self, value):
        if value is None:
            terminals = np.arange(self.n_terminals)
        elif not isinstance(value, np.ndarray):
            terminals = [v for v in value]    # this is to convert list of strings to chararray
        
        if len(terminals) != self.n_terminals:
            raise ValueError('terminal nodes size mismatch: %d instead of %d'
                                %(len(terminals), self.n_terminals))

        self._terminals = np.asarray(terminals)

    def get_terminals(self):
        return self._terminals

    terminals = property(get_terminals, set_terminals, 
                          doc='terminals nodes')

    def get_assignment(self):
        return self._assignment

    assignment = property(get_assignment, doc='assignment matrix')

    def relabel(self):
        mapping = {}
        map_indices = defaultdict(int)
        if self.assume_levels:
            map_dict = self.level()
        else:
            map_dict = self.depth()
            
        for node in map_dict:
            if istuple(node):
                value = map_dict[node]
                idx = map_indices[value]
                mapping[node] = (value, idx)

                map_indices[value] += 1

        self.hier = nx.relabel_nodes(self.hier, mapping, copy=True)
        return mapping

    def get_levels(self):
        # """Returns the levels (ordered ascendingly) in the hierarchy."""

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        T = self.hier
        levels = []

        for node in internals(T): 
            level = T.nodes[node]['level']
            if level not in levels:
                levels.append(level)

        levels.sort()
        return levels

    # NEW
    # def optimal_disjoint_nodes(self, feature='index', min_size=4, mode=''):
    #     '''
    #
    #     :return:
    #     '''
    #     if self.hier is None:
    #         raise ValueError('hierarchy not built. Call weave() first')
    #
    #     T = self.hier
    #     # nodesAt5 = [x for x, y in T.nodes(data=True) if y['feature'] == 5]
    #
    #     # select high stability and avoid parent and children. Have a minimum, and have a tolerance


    def some_node_from_level(self, level):
        # """Returns the first node that is associated with the partition specified by level."""

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        T = self.hier

        for node in internals(T):
            if T.nodes[node]['level'] == level:
                return node


    def weave(self, partitions, terminals=None, boolean=True, levels=False, **kwargs):
        """Finds a directed acyclic graph that represents a hierarchy recovered from 
        partitions.

        Parameters
        ----------
        partitions : positional argument 
            a list of different partitions of the graph. Each item in the 
            list should be an array (Numpy array or list) of partition labels 
            for the nodes. A root partition (where all nodes belong to one 
            cluster) and a terminal partition (where all nodes belong to 
            their own cluster) will automatically added later.

        terminals : keyword argument, optional (default=None)
            a list of names for the graph nodes. If none is provided, an 
            integer will be assigned to each node.

        levels : keyword argument, optional (default=False)
            whether assume the partitions are provided in some order. If set 
            to True, the algorithm will only find the parents for a node from 
            upper levels. The levels are assumed to be arranged in an ascending 
            order, e.g. partitions[0] is the highest level and partitions[-1] 
            the lowest.

        boolean : keyword argument, optional (default=False)
            whether the partition labels should be considered as boolean. If 
            set to True, only the clusters labelled as True will be considered 
            as a parent in the hierarchy.
        
        merge : keyword argument, optional (default=False)
            whether merge similar clusters. if one cluster is contained in another cluster (determined by "cutoff" oarameter) and vice versa, these two clusters deemed to be very similar. if set to true, such clusters groups will be merged into one (take union)
        
        top : keyword argument (0 ~ 100, default=100)
            top x percent (alternative) edges to be kept in the hierarchy. This parameter 
            controls the number of parents each node has based on a global ranking of the 
            edges. Note that if top=0 then each node will only have exactly one parent 
            (except for the root which has none). 

        cutoff : keyword argument (0.5 ~ 1.0, default=0.75)
            containment index cutoff for claiming parenthood.

        Returns
        -------
        T : networkx.DiGraph
            
        """

        top = kwargs.pop('top', 100)

        ## checkers
        n_sets = len(partitions)
        if n_sets == 0:
            raise ValueError('partitions cannot be empty')

        lengths = set([len(l) for l in partitions])
        if len(lengths) > 1:
            raise ValueError('partitions must have the same length')
        n_nodes = lengths.pop()

        if not isinstance(partitions, np.ndarray):
            arr = [[None]*n_nodes for _ in range(n_sets)] # ndarray(object) won't treat '1's correctly
            for i in range(n_sets):
                for j in range(n_nodes):
                    arr[i][j] = partitions[i][j]  # this is to deal with list-of-strings case
        else:
            arr = partitions
        partitions = arr 

        ## initialization
        if isinstance(levels, bool):
            self.assume_levels = levels
            levels = None
        else:
            self.assume_levels = True
        clevels = levels if levels is not None else np.arange(n_sets)
        if len(clevels) != n_sets:
            raise ValueError('levels/partitions length mismatch: %d/%d'%(len(levels), n_sets))

        if boolean:
            # convert partitions twice for CI calculation
            self._assignment = np.asarray(partitions).astype(bool)  # asarray(partitions, dtype=bool) won't treat '1's correctly
            self._labels = np.ones(n_sets, dtype=bool)
            self._levels = clevels
            #TODO: maybe use levels information to store resolution in finder, so comparison in weaver step can be shortened
        else:
            L = []; indices = []; labels = []; levels = []
            for i, p in enumerate(partitions):
                p = np.asarray(p)
                for label in np.unique(p):
                    indices.append(i)
                    labels.append(label)
                    levels.append(clevels[i])
                    L.append(p == label)
            self._assignment = np.vstack(L)
            self._labels = np.array(labels)
            self._levels = np.array(levels)

        self.terminals = terminals  # if terminal is None, a default array will be created by the setter

        ## build tree
        self._build(**kwargs)

        ## pick parents
        T = self.pick(top)

        return T

    def _build(self, **kwargs):
        """Finds all the direct parents for the clusters in partitions. This is the first 
        step of weave(). Subclasses can override this function to achieve different results.

        Parameters
        ----------
        cutoff : keyword argument (0.5 ~ 1.0, default=0.75)
            containment index cutoff for claiming parenthood.
        merge : bool

        Returns
        -------
        G : networkx.DiGraph
            
        """

        cutoff = kwargs.pop('cutoff', 0.75)
        merge = kwargs.pop('merge', False)

        assume_levels = self.assume_levels
        terminals = self.terminals
        n_nodes = self.n_terminals
        L = self._assignment
        labels = self._labels
        levels = self._levels

        n_sets = len(L)

        rng = range(n_sets)
        if assume_levels:
            gen = ((i, j) for i, j in iproduct(rng, rng) if levels[i] > levels[j])
        else:
            gen = ((i, j) for i, j in iproduct(rng, rng) if i != j)
    
        # find all potential parents
        LOGGER.timeit('_init')
        LOGGER.info('initializing the graph...')
        # calculate containment indices
        CI = containment_indices_boolean(L, L)
        G = nx.DiGraph()
        for i, j in gen:
            C = CI[i, j]
            na = (i, 0)
            nb = (j, 0)

            if not G.has_node(na):
                G.add_node(na, index=i, level=levels[i], label=labels[i])

            if not G.has_node(nb):
                G.add_node(nb, index=j, level=levels[j], label=labels[j])

            if C >= cutoff:
                if not merge:
                    if G.has_edge(na, nb):
                        C0 = G[na][nb]['weight']
                        if C > C0:
                            G.remove_edge(na, nb)
                        else:
                            continue
                G.add_edge(nb, na, weight=C)

        LOGGER.report('graph initialized in %.2fs', '_init')

        # remove loops
        if merge:

            def _collapse_nodes(G, vs):
                all_in_nodes, all_out_nodes = [], []
                vs = list(vs)
                vs = sorted(vs, key = lambda x:G.nodes[x]['index'])

                # add merge record
                new_index = []
                for v in vs:
                    new_index.append(G.nodes[v]['index'])
                    all_in_nodes.extend([w for w in G.predecessors(v)])
                    all_out_nodes.extend([w for w in G.successors(v)])
                all_in_nodes = list(set(all_in_nodes).difference(vs))
                all_out_nodes = list(set(all_out_nodes).difference(vs))
                dict_in_weights = {u:0 for u in all_in_nodes}
                dict_out_weights = {u:0 for u in all_out_nodes}
                for v in vs:
                    for w in G.predecessors(v):
                        if not w in all_in_nodes:
                            continue
                        if G[w][v]['weight'] > dict_in_weights[w]:
                            dict_in_weights[w] = G[w][v]['weight']
                for v in vs:
                    for w in G.successors(v):
                        if not w in all_out_nodes:
                            continue
                        if G[v][w]['weight'] > dict_out_weights[w]:
                            dict_out_weights[w] = G[v][w]['weight']

                G.remove_nodes_from(vs[1:])
                G.nodes[vs[0]]['index'] = tuple(new_index) # TODO: why does this has to be a tuple?
                for u in all_in_nodes:
                    if not G.has_predecessor(vs[0], u):
                        G.add_edge(u, vs[0], weight=dict_in_weights[u])
                for u in all_out_nodes:
                    if not G.has_successor(vs[0], u):
                        G.add_edge(vs[0], u, weight=dict_out_weights[u])

                return G

            LOGGER.timeit('_cluster_redundancy')
            LOGGER.info('"merge" parameter set to true, so merging redundant clusters...')

            try:
                cycles = list(nx.simple_cycles(G))
                # LOGGER.info('Merge {} redundant groups ...'.format(len(cycles)))
            except:
                LOGGER.info('No redundant groups has been found ...')
                cycles = []
            if len(cycles) > 0:
                Gcyc = nx.Graph()
                for i in range(len(cycles)):
                    for v, w in itertools.combinations(cycles[i], 2):
                        Gcyc.add_edge(v, w)
                components = list(nx.connected_components(Gcyc))
                LOGGER.info('Merge {} redundant groups ...'.format(len(components)))
                for vs in components:
                    G = _collapse_nodes(G, vs, )

            LOGGER.report('redundant nodes removed in %.2fs', '_cluster_redundancy')

        # add a root node to the graph
        roots = []
        for node, indeg in G.in_degree():
            if indeg == 0:
                roots.append(node)

        # TODO: right now needs a cluster full of 1, otherwise report an error; figure out why and fix it
        if len(roots) > 1:
            root = (-1, 0)  # (-1, 0) will be changed to (0, 0) later
            G.add_node(root, index=-1, level=-1, label=True)
            for node in roots:
                G.add_edge(root, node, weight=1.)
        else:
            root = roots[0]

        # remove grandparents (redundant edges)
        LOGGER.timeit('_redundancy')
        LOGGER.info('removing redudant edges...')
        redundant = []

        for node in G.nodes():
            parents = [_ for _ in G.predecessors(node)]
            ancestors = [_ for _ in nx.ancestors(G, node)]

            for a in parents:
                for b in ancestors:
                    if neq(a, b) and G.has_edge(a, b):
                        # a is a grandparent
                        redundant.append((a, node))
                        break

        G.remove_edges_from(redundant)
        LOGGER.report('redundant edges removed in %.2fs', '_redundancy')


        # attach graph nodes to nodes in G (this part can be skipped)

        LOGGER.timeit('_attach')
        LOGGER.info('attaching terminal nodes to the graph...')
        X = np.arange(n_nodes)
        nodes = [node for node in G.nodes]
        attached_record = defaultdict(list)

        for node in nodes:
            n = node[0]
            if n == -1:
                continue

            x = X[L[n]]

            for i in x:
                ter = denumpize(terminals[i])
                attached = attached_record[ter]
                
                skip = False
                if attached:
                    for other in reversed(attached):
                        if nx.has_path(G, node, other): # other is a descendant of node, skip
                            skip = True; break
                        elif nx.has_path(G, other, node): # node is a descendant of other, remove other
                            attached.remove(other)
                            G.remove_edge(other, ter)
                    
                if not skip:
                    G.add_edge(node, ter, weight=1.)
                    attached.append(node)

        LOGGER.report('terminal nodes attached in %.2fs', '_attach')

        # update node assignments
        LOGGER.timeit('_update')
        LOGGER.info('propagate terminal node assignments upward in the hierarchy') # TODO: this can be iterated until there's no change
        L_sp = sp.sparse.csr_matrix(L.T)

        ## construct a community connectivity matrix
        row, col = [], []
        for v in nodes:
            row.append(v[0])
            col.append(v[0])
        for v, w in itertools.combinations(nodes, 2):
            if nx.has_path(G, v, w): # w is a descendant of v
                row.append(w[0])
                col.append(v[0])
        data = np.ones_like(row, dtype=int)
        cc_mat = sp.sparse.coo_matrix((data, (row, col)), shape=(L.shape[0], L.shape[0]))
        cc_mat = cc_mat.tocsr()
        L_sp_new = L_sp.dot(cc_mat) > 0
        self._assignment = L_sp_new.toarray().T
        LOGGER.report('terminal nodes propagated in %.2fs', '_update')


        in_degrees = np.array([deg for (_, deg) in G.in_degree()])
        if np.where(in_degrees==0)[0] > 1:
            G.add_node('root') # add root
            # print('root added')
            for node in G.nodes():
                if G.in_degree(node)== 0:
                    G.add_edge('root', node)

        self._full = G
        
        # find secondary edges
        def node_size(node):
            if istuple(node):
                i = node[0]
                return np.count_nonzero(L[i])
            else:
                return 1

        LOGGER.timeit('_sec')
        LOGGER.info('finding secondary edges...')
        secondary = []
        for node in G.nodes():
            parents = [_ for _ in G.predecessors(node)]
            if len(parents) > 1:
                nsize = node_size(node)
                pref = []
                for p in parents:
                    w = G.edges()[p, node]['weight'] 
                    psize = node_size(p)
                    usize = w * nsize
                    j = usize / (nsize + psize - usize)
                    pref.append(j)

                # weight (CI) * node_size gives the size of the union between the node and the parent
                ranked_edges = [((x[0], node), x[1]) for x in sorted(zip(parents, pref), 
                                            key=lambda x: x[1], reverse=True)]
                secondary.extend(ranked_edges[1:])

        secondary.sort(key=lambda x: x[1], reverse=True)

        self._secondary = secondary
        LOGGER.report('secondary edges found in %.2fs', '_sec')

        return G

    def pick(self, top):
        """Picks top x percent edges. Alternative edges are ranked based on the number of 
        overlap terminal nodes between the child and the parent. This is the second 
        step of weave(). Subclasses can override this function to achieve different results.

        Parameters
        ----------
        top : int or float (0 ~ 100, default=100)
            top x percent (alternative) edges to be kept in the hierarchy. This parameter 
            controls the number of parents each node has based on a global ranking of the 
            edges. Note that if top=0 then each node will only have exactly one parent 
            (except for the root which has none). 

        Returns
        -------
        networkx.DiGraph
            
        """

        if self._secondary is None:
            raise ValueError('hierarchy not built. Call weave() first')

        G = self._full.copy()

        secondary = [x[0] for x in self._secondary]

        if top == 0:
            # special treatment for one-parent case for better performance
            G.remove_edges_from(secondary)
        elif top < 100:
            n = int(len(secondary) * top/100.)
            removed_edges = secondary[n:]
            G.remove_edges_from(removed_edges)


        # self.hier = prune(G)
        self.hier = G

        # update attributes
        self.update_depths()
        self.relabel()
        
        return self.hier

    # NEW
    def delete_nodes(self, nodes, relabel=False):
        '''
        Delete some nodes from the hierarchy. This approach can be used to delete those with low persistence and rebuild a simpler hierarchy.

        Parameters
        ----------
        nodes :  a list of string
            names of the cluster to delete.
        relabel : bool (default = False)
            if True, rename nodes. Setting to False may be easier to track the cluster identities.
        '''
        G = self.hier
        if G is None:
            raise ValueError('hierarchy not built. Call weave() first')
        for u in nodes:
            all_in_nodes = G.predecessors(u)
            all_out_nodes = G.successors(u)
            new_edge_pairs = [(a, b) for a, b in itertools.product(all_in_nodes, all_out_nodes)]
            G.add_edges_from(new_edge_pairs) # TODO: right now edges store some information about containment index, not seen from here
            G.remove_node(u)
        self.hier = G

        self.update_depths()
        if relabel:
            self.relabel()

        return

    # def select_nodes():



    def get_root(self):
        G = self.hier
        if G is None:
            raise ValueError('hierarchy not built. Call weave() first')

        return get_root(G)

    root = property(get_root, 'the root node')   
    
    def update_depths(self):
        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        T = self.hier

        def _update_topdown(parent):
            Q = [parent]

            while Q:
                parent = Q.pop(0)
                par_depth = T.nodes[parent]['depth']

                for child in T.successors(parent):
                    if 'depth' in T.nodes[child]:  # visited
                        ch_depth = T.nodes[child]['depth']
                        if ch_depth <= par_depth + 1:  # already shallower
                            continue
                    
                    T.nodes[child]['depth'] = par_depth + 1
                    Q.append(child)

        # update depths topdown
        root = self.root
        T.nodes[root]['depth'] = 0
        _update_topdown(root)

    def get_attribute(self, attr, node=None):
        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')
            
        G = self.hier

        # obtain values
        values = nx.get_node_attributes(G, attr)

        # pack results
        if node is None:
            ret = values
        elif istuple(node) or np.isscalar(node):
            ret = values.pop(node)
        else:
            ret = []
            for n in node:
                value = values[n] if n in values else None
                ret.append(value)

        return ret

    def depth(self, node=None):

        return self.get_attribute('depth', node)

    def level(self, node=None):

        return self.get_attribute('level', node)

    def get_max_depth(self):
        depths = self.depth(self.terminals)

        return np.max(depths)

    maxdepth = property(get_max_depth, 'the maximum depth of nodes to the root')   

    def all_depths(self, leaf=True):
        depth_dict = self.depth()

        if leaf:
            depths = [depth_dict[k] for k in depth_dict]
        else:
            depths = [depth_dict[k] for k in depth_dict if istuple(k)]

        return np.unique(depths)

    def show(self, **kwargs):
        """Visualize the hierarchy using networkx/graphviz hierarchical layouts.

        See Also
        --------
        show_hierarchy
            
        """

        nodelist = kwargs.pop('nodelist', None)
        dummy = kwargs.pop('dummy', False)

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        T = self.hier

        if nodelist is None:
            nodelist = T.nodes()

        if dummy:
            T = stuff_dummies(T)
        
        return show_hierarchy(T, nodelist=nodelist, **kwargs)

    def node_cluster(self, node, out=None): # TODO: this is useful, but hope to test it
        """Recovers the cluster represented by a node in the hierarchy.

        Returns
        -------
        H : a Numpy array of labels for all the terminal nodes.
            
        """

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        T = self.hier

        nodes = self.terminals
        n_nodes = self.n_terminals

        if out is None:
            out = np.zeros(n_nodes, dtype=bool)
        
        desc = [_ for _ in nx.descendants(T, node) 
                if not istuple(_)]
        
        if len(desc):
            for d in desc:
                j = find(nodes, d)

                out[j] = True
        else:
            j = find(nodes, node)
            out[j] = True

        return out

    def _topdown_cluster(self, attr, value, flat=True):
        """Recovers the partition at specified depth.

        Returns
        -------
        H : a Numpy array of labels for all the terminal nodes.
            
        """

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        T = self.hier
        n_nodes = self.n_terminals

        attrs = nx.get_node_attributes(T, attr)
        root = self.root

        # assign labels
        Q = [root]
        clusters = []
        visited = defaultdict(bool)

        while Q:
            node = Q.pop(0)

            if visited[node]:
                continue

            visited[node] = True

            if not istuple(node):
                clusters.append(node)
                continue

            if attrs[node] < value:
                for child in T.successors(node):
                    Q.append(child)
            elif attrs[node] == value:
                clusters.append(node)
            else:
                LOGGER.warn('something went wrong: visiting node with '
                            '%s greater than %d'%(attr, value))

        H = np.zeros((len(clusters), n_nodes), dtype=bool)

        for i, node in enumerate(clusters):
            self.node_cluster(node, H[i, :])

        if flat:
            I = np.arange(len(clusters)) + 1
            I = np.atleast_2d(I)
            H = H * I.T
            h = H.max(axis=0)
            return h
        
        return H

    def depth_cluster(self, depth, flat=True):
        """Recovers the partition at specified depth.

        Returns
        -------
        H : a Numpy array of labels for all the terminal nodes.
            
        """
        
        return self._topdown_cluster('depth', depth, flat)

    def level_cluster(self, level, flat=True):
        """Recovers the partition that specified by the index based on the 
        hierarchy.

        Returns
        -------
        H : a Numpy array of labels for all the terminal nodes.
            
        """

        if not self.assume_levels:
            LOGGER.warn('Levels were not followed when building the hierarchy.')

        return self._topdown_cluster('level', level, flat)

    def write(self, filename, format='ddot'):
        """Writes the hierarchy to a text file.

        Parameters
        ----------
        filename : str
            the path and name of the output file.

        format : str
            output format. Available options are "ddot".
            
        """

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        G = self.hier.copy()
        
        if format == 'ddot':
            for u, v in G.edges():
                if istuple(u) and istuple(v):
                    G[u][v]['type'] = 'Child-Parent'
                else:
                    G[u][v]['type'] = 'Gene-Term'

        mapping = {}
        for node in internals(G):
            strnode = '%s_%s'%node
            mapping[node] = strnode

        G = nx.relabel_nodes(G, mapping, copy=False)

        with open(filename, 'w') as f:
            f.write('Parent\tChild\tType\n')
        
        with open(filename, 'ab') as f:
            nx.write_edgelist(G, f, delimiter='\t', data=['type'])


def all_equal(iterable):
    #Returns True if all the elements are equal to each other

    from itertools import groupby

    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def boolize(x):
    #Converts x to boolean

    if not isinstance(x, bool):
        x = bool(int(x))
    return x

def neq(a, b):
    # workaround for a (potential) 
    # networkx bug related to numpy.int, 
    # e.g. (1, 2) == numpy.int32(1)

    r = a != b
    try:
        len(r)
    except TypeError:
        return True
    return r

def n_simple_paths(G, u, v):
    nsp_reg = {}

    def nsp(u, v):
        if u == v:
            return 1
        if not u in nsp_reg:
            npaths = 0
            for c in G.successors(u):
                npaths += nsp(c, v)
            nsp_reg[u] = npaths

        return nsp_reg[u]

    return nsp(u, v)

def denumpize(x):
    if isinstance(x, np.generic):
        return x.item()
    return x

def find(A, a):
    if isinstance(A, list):
        return A.index(a)
    else:
        A = np.asarray(A)
        return np.where(A==a)[0][0]

def get_root(T):
    for node, indeg in T.in_degree():
        if indeg == 0:
            return node

    return None

def prune(T): # TODO: this can be deprecated
    '''
    Removes the nodes with only one child and the nodes that have no terminal
    nodes (e.g. genes) as descendants. (This basically removes identical clusters)

    Parameters
    ----------
    T: a weaver object
    '''

    # prune tree
    # remove dead-ends
    # internal_nodes = [node for node in T.nodes() if istuple(node)]
    # out_degrees = [val for key, val in T.out_degree(internal_nodes)]

    # while (0 in out_degrees):
    #     for node in reversed(internal_nodes):
    #         outdeg = T.out_degree(node)
    #         if istuple(node) and outdeg == 0:
    #             T.remove_node(node)
    #             internal_nodes.remove(node)
    #
    #     out_degrees = [val for key, val in T.out_degree(internal_nodes)]

    # remove single branches
    def _single_branch(node):
        indeg = T.in_degree(node)
        outdeg = T.out_degree(node)

        if indeg > 1 or indeg == 0:
            return False
        
        if outdeg > 1 or outdeg == 0:
            return False

        # if attached to a terminal node, not considered as a single branch 
        child = next(T.successors(node))
        if not istuple(child):
            return False
        return True

    #all_nodes = [node for node in T.nodes()]
    all_nodes = [node for node in traverse_topdown(T)]

    for node in all_nodes:
        if _single_branch(node):
            parent = next(T.predecessors(node))
            child  = next(T.successors(node))
            
            w1 = T[parent][node]['weight']
            w2 = T[node][child]['weight']

            T.remove_node(node)
            T.add_edge(parent, child, weight=w1 + w2)

    return T

def traverse_topdown(T, mode='breadth'):
    if mode == 'depth':
        q = -1
    elif mode == 'breadth':
        q = 0
    else:
        raise ValueError('mode must be either "depth" or "breadth"')

    root = get_root(T)
    Q = [root]
    visited = defaultdict(bool)
    while Q:
        node = Q.pop(q)

        if visited[node]:
            continue
        
        visited[node] = True

        for child in T.successors(node):
            Q.append(child)
        yield node

def stuff_dummies(hierarchy):
    # """Puts dummy nodes into the hierarchy. The dummy nodes are used
    # in `show_hierarchy` when `assume_level` is True.
    #
    # Returns
    # -------
    # T : networkx.DiGraph
    #     An hierarchy with dummy nodes added.
    #
    # Raises
    # ------
    # ValueError
    #     If hierarchy has not been built.
    #
    # """

    T = hierarchy.copy()

    level_dict = nx.get_node_attributes(T, 'level')
    levels = np.unique(list(level_dict.values()))
    d = -1

    # make a list of node refs for the iteration during which nodes will change
    internal_nodes = [node for node in internals(T) if T.in_degree(node)]

    for node in internal_nodes:
        level = T.nodes[node]['level']
        i = find(levels, level)
        parents = [_ for _ in T.predecessors(node)]
        for parent in parents:
            if not istuple(parent):
                # unlikely to happen
                continue
            plevel = T.nodes[parent]['level']
            j = find(levels, plevel)
            n = i - j
            if n > 1:
                # add n-1 dummies
                T.remove_edge(parent, node)
                #d0 = None
                for i in range(1, n):
                    d += 1
                    l = levels[j + i]
                    #labels = [n[1] for n in T.nodes() if istuple(n) if T.nodes[n]['level']==l]
                    #d = getSmallestAvailable(labels)
                    
                    curr = (l, d, None)
                    if i == 1:
                        T.add_edge(parent, curr, weight=1)
                    else:
                        l0 = levels[j + i - 1]
                        T.add_edge((l0, d-1, None), curr, weight=1)
                    #d0 = d
                    T.nodes[curr]['level'] = l
                    #T.nodes[curr]['level'] = None

                T.add_edge(curr, node, weight=1)
    
    return T


# TODO: make this using dash
def show_hierarchy(T, **kwargs):
    #TODO: dependency here is not declared
    #TODO: node label?
    """Visualizes the hierarchy in notebook"""

    from networkx.drawing.nx_pydot import write_dot, graphviz_layout
    from networkx import draw, get_edge_attributes, draw_networkx_edge_labels
    from os import name as osname

    from matplotlib.pyplot import plot, xlim, ylim

    style = kwargs.pop('style', 'dot')
    leaf = kwargs.pop('leaf', True)
    nodesize = kwargs.pop('node_size', 16)
    edgescale = kwargs.pop('edge_scale', None)
    edgelabel = kwargs.pop('edge_label', False)
    interactive = kwargs.pop('interactive', True)

    isWindows = osname == 'nt'

    if isWindows:
        style += '.exe'

    if not leaf: # TODO: leaf false has a bug
        T2 = T.subgraph(n for n in T.nodes() if istuple(n))
        if 'nodelist' in kwargs:
            nodes = kwargs.pop('nodelist')
            nonleaves = []
            for node in nodes:
                # if istuple(node): # TODO: fix here
                nonleaves.append(node)
            kwargs['nodelist'] = nonleaves
    else:
        T2 = T
    
    pos = graphviz_layout(T2, prog=style)

    if edgescale:
        widths = []
        for u, v in T2.edges():
            w = T2[u][v]['weight']*edgescale
            widths.append(w) 
    else:
        widths = 1.

    if not 'arrows' in kwargs:
        kwargs['arrows'] = False
        
    draw(T2, pos, node_size=nodesize, width=widths, **kwargs)
    if edgelabel:
        labels = get_edge_attributes(T2, 'weight')
        draw_networkx_edge_labels(T2, pos, edge_labels=labels)

    if interactive:
        from matplotlib.pyplot import gcf
        from scipy.spatial.distance import cdist

        annotations = {}

        def _onclick(event):
            ax = event.inaxes
            if ax is not None:
                if event.button == 1:
                    view = ax.viewLim.extents
                    xl = (view[0], view[2])
                    yl = (view[1], view[3])
                    xl, yl = ax.get_xlim(), ax.get_ylim()
                    x_, y_ = event.xdata, event.ydata

                    dx = (xl[1] - xl[0]) / 20 
                    dy = (yl[1] - yl[0]) / 20
                    dr = min((dx, dy))

                    nodes = []; coords = []
                    for node, coord in pos.items():
                        nodes.append(node)
                        coords.append(coord)

                    D = cdist([(x_, y_)], coords).flatten()
                    i = D.argmin()

                    if D[i] < dr:
                        x, y = coords[i]
                        node = nodes[i]
                        
                        if node not in annotations:
                            #ax.plot([i, i], ax.get_ylim(), 'k--')
                            l = ax.plot([x], [y], 'bo', fillstyle='none', markersize=nodesize//2)
                            t = ax.text(x, y, str(node), color='k')
                            annotations[node] = (l[0], t)
                            xlim(xl); ylim(yl)
                        else:
                            l, t = annotations.pop(node)
                            ax.lines.remove(l)
                            ax.texts.remove(t) 
                elif event.button == 3:
                    for node in annotations:
                        l, t = annotations[node]
                        ax.lines.remove(l)
                        ax.texts.remove(t)
                    annotations.clear()
                fig.canvas.draw()

        fig = gcf()
        cid = fig.canvas.mpl_connect('button_press_event', _onclick)

    return T2, pos