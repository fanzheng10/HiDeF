import numpy as np
import scipy as sp
import networkx as nx
from collections import Counter, defaultdict

from hierweaver import LOGGER

__all__ = ['Weaver']

istuple = lambda n: isinstance(n, tuple)
isdummy = lambda n: None in n
internals = lambda T: (node for node in T if istuple(node))

class Weaver(object):
    """
    Class for constructing a hierarchical representation of a graph 
    based on (a list of) input partitions. 

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

    assume_levels : keyword argument, optional (default=False)
        whether assume the partitions are provided in some order. If set 
        to True, the algorithm will only find the parents for a node from 
        upper levels. The levels are assumed to be arranged in an ascending 
        order, e.g. partitions[0] is the highest level and partitions[-1] 
        the lowest.

    boolean : keyword argument, optional (default=False)
        whether the partition labels should be considered as boolean. If 
        set to True, only the clusters labelled as True will be considered 
        as a parent in the hierarchy.

    Examples
    --------
    Define a list of partitions P:

    >>> P = ['11111111',
    ...      '11111100',
    ...      '00001111',
    ...      '11100000',
    ...      '00110000',
    ...      '00001100',
    ...      '00000011']

    Define terminal node labels:

    >>> nodes = 'ABCDEFGH'

    Construct a hierarchy based on P:

    >>> weaver = Weaver(P, boolean=True, terminals=nodes, assume_levels=False)
    >>> T = weaver.weave(cutoff=0.9, top=10)

    """

    __slots__ = ['_assignment', '_terminals', 'assume_levels', 'hier', '_levels', '_labels',
                 '_dhier', '_full', '_secondary']

    def __init__(self):
        self.hier = None
        self._dhier = None
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
        depth_dict = self.depth()
        mapping = {}
        depth_indices = defaultdict(int)

        for node in depth_dict:
            if istuple(node):
                depth = depth_dict[node]
                idx = depth_indices[depth]
                mapping[node] = (depth, idx)

                depth_indices[depth] += 1

        self.hier = nx.relabel_nodes(self.hier, mapping, copy=True)
        return mapping

    def stuff_dummies(self):
        """Puts dummy nodes into the hierarchy. The dummy nodes are used 
        in level_cluster() and show() when assume_level is True.

        Returns
        -------
        T : networkx.DiGraph
            An hierarchy with dummy nodes added.

        Raises
        ------
        ValueError
            If hierarchy has not been built.

        """

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        T = self.hier.copy()

        levels = self.get_levels()
        d = -1

        # make a list of node refs for the iteration during which nodes will change
        internal_nodes = [node for node in internals(T) if T.in_degree(node)]

        for node in internal_nodes:
            level = T.nodes[node]['level']
            i = levels.index(level)
            parents = [_ for _ in T.predecessors(node)]
            for parent in parents:
                if not istuple(parent):
                    # unlikely to happen
                    continue
                plevel = T.nodes[parent]['level']
                j = levels.index(plevel)
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
        
        self._dhier = T
        return T

    def get_levels(self, ignore_dummies=True):
        """Returns the levels (ordered ascendingly) in the hierarchy."""

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        T = self.hier
        levels = []

        if ignore_dummies:
            internal_nodes = (node for node in internals(T) if not isdummy(node))
        else:
            internal_nodes = internals(T)
        for node in internal_nodes: 
            level = T.nodes[node]['level']
            if level not in levels:
                levels.append(level)

        levels.sort()
        return levels

    def some_node(self, level):
        """Returns the first node that is associated with the partition specified by level."""

        if self.assume_levels:
            T = self.hier
        else:
            T = self._dhier

        for node in internals(T):
            if T.nodes[node]['level'] == level:
                return node

    def weave(self, partitions, terminals=None, assume_levels=False, 
                 boolean=False, levels=None, **kwargs):
        """Finds a directed acyclic graph that represents a hierarchy recovered from 
        partitions.

        Parameters
        ----------
        top : keyword argument (0 ~ 100, default=100)
            top x percent (alternative) edges to be kept in the hierarchy. This parameter 
            controls the number of parents each node has based on a global ranking of the 
            edges. Note that if top=0 then each node will only have exactly one parent 
            (except for the root which has none). 

        cutoff : keyword argument (0.5 ~ 1.0, default=0.8)
            containment index cutoff for claiming parenthood. c

        See Also
        --------
        build
        pick

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
        clevels = levels if levels is not None else np.arange(n_sets)
        if len(clevels) != n_sets:
            raise ValueError('levels/partitions length mismatch: %d/%d'%(len(levels), n_sets))

        if boolean:
            # convert partitions twice for CI calculation
            self._assignment = np.asarray(partitions).astype(bool)  # asarray(partitions, dtype=bool) won't treat '1's correctly
            self._labels = np.ones(n_sets, dtype=bool)
            self._levels = clevels
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
        self.assume_levels = assume_levels

        ## build tree
        T = self._build(**kwargs)

        ## pick parents
        T = self.pick(top)

        return T

    def _build(self, **kwargs):
        """Finds all the direct parents for the clusters in partitions. This is the first 
        step of weave(). Subclasses can override this function to achieve different results.

        Parameters
        ----------
        cutoff : keyword argument (0.5 ~ 1.0, default=0.8)
            containment index cutoff for claiming parenthood. 

        Returns
        -------
        T : networkx.DiGraph
            
        """

        from itertools import product

        cutoff = kwargs.pop('cutoff', 0.8)

        assume_levels = self.assume_levels
        terminals = self.terminals
        n_nodes = self.n_terminals
        L = self._assignment
        labels = self._labels
        levels = self._levels

        n_sets = len(L)

        rng = range(n_sets)
        if assume_levels:
            gen = ((i, j) for i, j in product(rng, rng) if levels[i] > levels[j])
        else:
            gen = ((i, j) for i, j in product(rng, rng) if i != j)
    
        # find all potential parents
        LOGGER.timeit('_init')
        LOGGER.debug('initializing the graph...')
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
                if (na, nb) in G.edges():
                    C0 = G[na][nb]['weight']
                    if C > C0:
                        G.add_edge(nb, na, weight=C)
                        G.remove_edge(na, nb)
                else:
                    G.add_edge(nb, na, weight=C)

        LOGGER.report('graph initialized in %.2fs', '_init')

        # remove grandparents (redundant edges)
        LOGGER.timeit('_redundancy')
        LOGGER.debug('removing redudant edges...')
        redundant = []
        for node in G.nodes():
            parents = [_ for _ in nx.ancestors(G, node)]

            for a in parents:
                for b in parents:
                    if neq(a, b):
                        if G.has_edge(a, b):
                            # a is a grandparent
                            redundant.append((a, node))
                            break
                        if G.has_edge(b, a):
                            # b is a grandparent
                            redundant.append((b, node))
        
        # for u, v in G.edges():
        #     if has_multiple_paths(G, u, v):
        #         redundant.append((u, v))

        G.remove_edges_from(redundant)
        LOGGER.report('redundant edges removed in %.2fs', '_redundancy')

        # add a root node to the graph
        roots = []
        for node, indeg in G.in_degree():
            if indeg == 0:
                roots.append(node)

        if len(roots) > 1:
            root = (-1, 0)   # (-1, 0) will be changed to (0, 0) later
            G.add_node(root, index=-1, level=-1, label=1)
            for node in roots:
                G.add_edge(root, node, weight=1.)
        else:
            root = roots[0]

        # attach graph nodes to nodes in G
        # for efficiency purposes, this is done after redundant edges are 
        # removed. So we need to make sure we don't introduce new redundancy
        LOGGER.timeit('_attach')
        LOGGER.debug('attaching terminal nodes to the graph...')
        X = np.arange(n_nodes)
        nodes = [node for node in G.nodes]
        attached_record = defaultdict(list)

        for node in nodes:
            n = node[0]
            if n == -1:
                continue

            x = X[L[n]]

            for i in x:
                ter = terminals[i]
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

        self._full = G
        
        # find secondary edges
        def node_size(node):
            if istuple(node):
                i = node[0]
                return np.count_nonzero(L[i])
            else:
                return 1

        LOGGER.timeit('_sec')
        LOGGER.debug('finding secondary edges...')
        secondary = []
        for node in G.nodes():
            parents = [_ for _ in G.predecessors(node)]
            if len(parents) > 1:
                nsize = node_size(node)
                #weights = [G.edges()[p, node]['weight'] for p in parents]

                # preference if multiple best
                # if self.assume_levels:
                #     pref = [G.nodes[p]['index'] for p in parents]
                # else:
                #     pref = []
                #     for p in parents:
                #         nsize = -self.node_size(p) # use negative size to sort in ascending order when reversed
                #         pref.append(nsize)
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
        top : keyword argument (0 ~ 100, default=100)
            top x percent (alternative) edges to be kept in the hierarchy. This parameter 
            controls the number of parents each node has based on a global ranking of the 
            edges. Note that if top=0 then each node will only have exactly one parent 
            (except for the root which has none). 

        Returns
        -------
        T : networkx.DiGraph
            
        """

        if self._secondary is None:
            raise ValueError('hierarchy not built. Call weave() first')

        G = self._full.copy()

        #W = [x[1] for x in self._secondary]
        secondary = [x[0] for x in self._secondary]

        if top == 0:
            # special treatment for one-parent case for better performance
            G.remove_edges_from(secondary)
        elif top < 100:
            n = int(len(secondary) * top/100.)
            removed_edges = secondary[n:]
            G.remove_edges_from(removed_edges)

        # prune tree
        T = prune(G)

        self.hier = self._dhier = T

        # update attributes
        self.update_depths()

        if self.assume_levels:
            self.stuff_dummies()
        else:
            self.relabel()
        return T

    def get_root(self):
        G = self.hier
        if G is None:
            raise ValueError('hierarchy not built. Call weave() first')

        for node, indeg in G.in_degree():
            if indeg == 0:
                return node

        return None

    root = property(get_root, 'the root node')   
    
    def update_depths(self):
        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        T = self.hier

        def _update_topdown(parent):
            par_depth = T.nodes[parent]['depth']

            children = T.successors(parent)
            for child in children:
                if 'depth' in T.nodes[child]:  # visited
                    ch_depth = T.nodes[child]['depth']
                    if ch_depth <= par_depth + 1:  # already shallower
                        continue

                T.nodes[child]['depth'] = par_depth + 1
                _update_topdown(child)

        # update depths topdown
        root = self.root
        T.nodes[root]['depth'] = 0
        _update_topdown(root)

    def depth(self, node=None):
        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')
            
        G = self.hier

        # obtain depths
        depths = nx.get_node_attributes(G, 'depth')

        # pack results
        if node is None:
            ret = depths
        elif istuple(node) or np.isscalar(node):
            ret = depths.pop(node)
        else:
            ret = []
            for n in node:
                ret.append(depths[n])

        return ret

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

        if self.assume_levels and dummy:
            T = self._dhier
        else:
            T = self.hier

        if nodelist is None:
            nodelist = self.hier.nodes()
        
        return show_hierarchy(T, nodelist=nodelist, **kwargs)

    def level_cluster(self, level):
        """Recovers the partition that specified by the index based on the 
        hierarchy.

        Returns
        -------
        H : a Numpy array of labels for all the terminal nodes.
            
        """

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        if self.assume_levels:
            T = self._dhier
        else:
            T = self.hier

        nodes = self.terminals
        n_nodes = self.n_terminals

        levels = nx.get_node_attributes(T, 'level')
        ancestors = [node for node in internals(T) if levels[node]==level]
                
        H = np.zeros(n_nodes, dtype=int)
        for i, node in enumerate(ancestors):
            desc = (_ for _ in nx.descendants(T, node) 
                    if not istuple(_))
            for d in desc:
                j = nodes.index(d)

                H[j] = i + 1

        return H

    def depth_cluster(self, depth):
        """Recovers the partition at specified depth.

        Returns
        -------
        H : a Numpy array of labels for all the terminal nodes.
            
        """

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        T = self.hier

        nodes = self.terminals
        n_nodes = self.n_terminals

        depths = self.depth()
        internal_nodes = [_ for _ in internals(T)]
        depth_nodes = [_ for _ in internal_nodes if depths[_]==depth]

        # assign labels
        H = np.zeros(n_nodes, dtype=int)
        for i, node in enumerate(depth_nodes):
            desc = (_ for _ in nx.descendants(T, node) 
                    if not istuple(_))
            for d in desc:
                j = nodes.index(d)

                if H[j] == 0:
                    H[j] = i + 1
        
        # find nodes with unassigned 
        i += 1
        sec_depth_nodes = {}
        for j, h in enumerate(H):
            if h:
                continue

            node = nodes[j]
            
            # pick a parent with the most depth
            parents = []
            par_depths = []
            for parent in T.predecessors(node):
                parents.append(parent)
                par_depths.append(depths[parent])

            p = np.argmax(par_depths)
            parent = parents[p]

            if parent not in sec_depth_nodes:
                sec_depth_nodes[parent] = H[j] = i + 1
                i += 1
            else:
                H[j] = sec_depth_nodes[parent]

        return H

    def node_cluster(self, node):
        """Recovers the cluster represented by a node in the hierarchy.

        Returns
        -------
        H : a Numpy array of labels for all the terminal nodes.
            
        """

        if self.hier is None:
            raise ValueError('hierarchy not built. Call weave() first')

        if node in self.hier.nodes:
            T = self.hier
        elif node in self._dhier.nodes:
            T = self._dhier
        else:
            raise ValueError('node %s not found in hierarchy'%str(node))

        nodes = self.terminals
        n_nodes = self.n_terminals

        H = np.zeros(n_nodes, dtype=bool)
        desc = [_ for _ in nx.descendants(T, node) 
                if not istuple(_)]
        
        if len(desc):
            for d in desc:
                j = nodes.index(d)

                H[j] = True
        else:
            j = nodes.index(node)
            H[j] = True

        return H

    def write(self, filename, format='ddot'):
        """Writes the hierarchy to a text file.

        Parameters
        ----------
        filename : the path and name of the output file.

        format : output format. Available options are "ddot".
            
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

def containment_indices_legacy(A, B):
    from collections import defaultdict

    n = len(A)
    counterA  = defaultdict(int)
    counterB  = defaultdict(int)
    counterAB = defaultdict(int)

    for i in range(n):
        a = A[i]; b = B[i]
        
        counterA[a] += 1
        counterB[b] += 1
        counterAB[(a, b)] += 1

    LA = [l for l in counterA]
    LB = [l for l in counterB]

    CI = np.zeros((len(LA), len(LB)))
    for i, a in enumerate(LA):
        for j, b in enumerate(LB):
            CI[i, j] = counterAB[(a, b)] / counterA[a]

    return CI, LA, LB

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
    count = np.count_nonzero(A, axis=1)

    A = A.astype(float)
    B = B.astype(float)
    overlap = A.dot(B.T)
    
    CI = overlap / count[:, None]
    return CI

def containment_indices_sparse(A, B, sparse=False):
    '''
    calculate containment index for all clusters in A in all clusters in B
    :param A: a numpy matrix, axis 0 - cluster; axis 1 - nodes
    :param B: a numpy matrix, axis 0 - cluster; axis 1 - nodes
    :return: a sparse matrix with containment index; calling row/column/data for individual pairs
    '''
    if not sparse:
        Asp = sp.sparse.csr_matrix(A)
        Bsp = sp.sparse.csr_matrix(B)
    else:
        Asp = A
        Bsp = B
    both = np.asarray(np.sum(Asp.multiply(Bsp), axis=1)).ravel()
    countA =  Asp.getnnz(axis=1) # this is dense matrix
    contain = 1.0 * both/countA
    # print(both, countA, contain)
    return contain

def all_equal(iterable):
    "Returns True if all the elements are equal to each other"

    from itertools import groupby

    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def boolize(x):
    "Converts x to boolean"

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
    queue = [(u, v)]
    nsp = 0
    
    while queue:
        a, b = queue.pop(0)
        if a == b:
            nsp += 1
        else:
            for c in G.successors(a):
                queue.append((c, b))
    return nsp

def has_multiple_paths(G, u, v):
    Q = [(u, v)]
    nsp = 0
    
    while Q:
        a, b = Q.pop(-1)
        if a == b:
            nsp += 1
        else:
            for c in G.successors(a):
                Q.append((c, b))
        
        if nsp >= 2:
            return True
    return False

def prune(T):
    """Removes the nodes with only one child and the nodes that have no terminal 
    nodes (e.g. genes) as descendants."""

    # prune tree
    # remove dead-ends
    internal_nodes = [node for node in T.nodes() if istuple(node)]
    out_degrees = [val for key, val in T.out_degree(internal_nodes)]

    while (0 in out_degrees):
        for node in reversed(internal_nodes):
            outdeg = T.out_degree(node)
            if istuple(node) and outdeg == 0:
                T.remove_node(node)
                internal_nodes.remove(node)

        out_degrees = [val for key, val in T.out_degree(internal_nodes)]

    # remove single branches
    all_nodes = [node for node in T.nodes()]
    for node in all_nodes:
        indeg = T.in_degree(node)
        outdeg = T.out_degree(node)

        if indeg == 1 and outdeg == 1:
            parent = next(T.predecessors(node))
            child  = next(T.successors(node))
            
            w1 = T[parent][node]['weight']
            w2 = T[node][child]['weight']

            T.remove_node(node)
            T.add_edge(parent, child, weight=w1 + w2)
    return T

def show_hierarchy(T, **kwargs):
    """Visualizes the hierarchy"""

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

    if not leaf:
        T2 = T.subgraph(n for n in T.nodes() if istuple(n))
        if 'nodelist' in kwargs:
            nodes = kwargs.pop('nodelist')
            nonleaves = []
            for node in nodes:
                if istuple(node):
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

def weave(partitions, terminals=None, **kwargs):
    weaver = Weaver()
    T = weaver.weave(partitions, terminals, **kwargs)

    return weaver

if __name__ == '__main__':
    from pylab import *

    # L = ['11111111',
    #      '11111100',
    #      '00001111',
    #      '11100000',
    #      '00110000',
    #      '00001100',
    #      '00000011']

    # L = [list(p) for p in L]
    # nodes = 'ABCDEFGH'

    # weaver = Weaver(L, boolean=True, terminals=nodes, assume_levels=False)
    # weaver.weave(cutoff=0.9, top=100)
    # weaver.pick(0)

    # ion()
    # figure()
    # weaver.show()

    P = ['11111111',
         '11111100',
         '00001111',
         '11100000',
         '00110000',
         '00001100',
         '00000011']
    P = [list(p) for p in P]

    nodes = 'ABCDEFGH'

    w = Weaver()
    T = w.weave(P, boolean=True, terminals=nodes, cutoff=0.9, top=10)
