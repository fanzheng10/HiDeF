import numpy as np
import networkx as nx

from collections import Counter, defaultdict

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

    __slots__ = ['_partitions', '_terminals', 'assume_levels', 'hier', 
                 '_dhier', '_full', 'boolean', '_secondary', 'clevels']

    def __init__(self, partitions, terminals=None, assume_levels=False, 
                 boolean=False, clevels=None):
        self.partitions = partitions
        self.terminals = terminals

        self.assume_levels = assume_levels
        self.hier = None
        self._dhier = None
        self.boolean = boolean
        self._full = None
        self._secondary = None
        if clevels != None:
            self.clevels = clevels
        else:
            self.clevels = [i for i in range(len(partitions))]

    def set_partitions(self, value):
        # checkers
        n_sets = len(value)
        if n_sets == 0:
            raise ValueError('partitions cannot be empty')
        
        lengths = set([len(l) for l in value])
        if len(lengths) > 1:
            raise ValueError('partitions must have the same length')
        n_nodes = lengths.pop()

        if not isinstance(value, np.ndarray):
            arr = np.empty((n_sets, n_nodes), dtype=object)
            for i in range(n_sets):
                for j in range(n_nodes):
                    arr[i, j] = value[i][j]
        else:
            arr = value

        self._partitions = arr
        self.hier = None
        self._dhier = None
        self._full = None
        self._secondary = None

    def get_partitions(self):
        return self._partitions

    partitions = property(get_partitions, set_partitions, 
                          doc='the list of partitions')

    def get_n_partitions(self):
        return len(self._partitions)

    n_partitions = property(get_n_partitions, 'the number of partitions')   

    def get_n_terminals(self):
        return len(self._partitions[0])

    n_terminals = property(get_n_terminals, 'the number of terminal nodes')    

    def set_terminals(self, value):
        if value is None:
            terminals = list(range(self.n_terminals))
        else:
            terminals = [v for v in value]
        
        if len(terminals) != self.n_terminals:
            raise ValueError('terminal nodes size mismatch: %d instead of %d'
                                %(len(terminals), self.n_terminals))

        if isinstance(terminals, np.ndarray):
            terminals = terminals.tolist()

        self._terminals = terminals

    def get_terminals(self):
        return self._terminals

    terminals = property(get_terminals, set_terminals, 
                          doc='terminals nodes')

    def node_size(self, node):
        """Puts dummy nodes into the hierarchy. The dummy nodes are used 
        in level_cluster() and show() when assume_level is True.

        Parameters
        ----------
        node : a node in the hierarchy.

        Returns
        -------
        s : the size of the node. If the node is non-terminal, then s will 
            be the number of terminal nodes that are descendants to it.
            
        """

        L = self._partitions
        T = self._full

        if istuple(node):
            s = 0
            n  = T.nodes[node]['index']
            ln = T.nodes[node]['label']
            # use iteration for data type compatibility
            for l in L[n]:
                if l == ln:
                    s += 1
        else:
            s = 1
        return s

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
            level = T.nodes[node]['index']
            i = levels.index(level)
            parents = [_ for _ in T.predecessors(node)]
            for parent in parents:
                if not istuple(parent):
                    # unlikely to happen
                    continue
                plevel = T.nodes[parent]['index']
                j = levels.index(plevel)
                n = i - j
                if n > 1:
                    # add n-1 dummies
                    T.remove_edge(parent, node)
                    #d0 = None
                    for i in range(1, n):
                        d += 1
                        l = levels[j + i]
                        #labels = [n[1] for n in T.nodes() if istuple(n) if T.nodes[n]['index']==l]
                        #d = getSmallestAvailable(labels)
                        
                        curr = (l, d, None)
                        if i == 1:
                            T.add_edge(parent, curr, weight=1)
                        else:
                            l0 = levels[j + i - 1]
                            T.add_edge((l0, d-1, None), curr, weight=1)
                        #d0 = d
                        T.nodes[curr]['index'] = l
                        T.nodes[curr]['label'] = None

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
            level = T.nodes[node]['index']
            if level not in levels:
                levels.append(level)

        levels.sort()
        return levels

    def some_node(self, index):
        """Returns the first node that is associated with the partition specified by index."""

        if self.assume_levels:
            T = self.hier
        else:
            T = self._dhier

        for node in internals(T):
            if T.nodes[node]['index'] == index:
                return node

    def weave(self, **kwargs):
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

        # build tree
        T = self._build(**kwargs)

        # pick parents
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
        L = self._partitions

        assume_levels = self.assume_levels
        boolean = self.boolean
        
        terminals = self.terminals
        n_nodes = self.n_terminals

        n_sets = len(L)

        rng = range(n_sets)
        if assume_levels:
            print(len(rng), len(self.clevels))
            gen = ((i, j) for i, j in product(rng, rng) if self.clevels[i] > self.clevels[j])
        else:
            gen = ((i, j) for i, j in product(rng, rng) if i != j)
    
        # find all potential parents
        G = nx.DiGraph()
        for i, j in gen:
            A = L[i]; B = L[j]
            CI, LA, LB = containment_indices(A, B)

            for a, la in enumerate(LA):
                if boolean and not boolize(la):
                    continue
                for b, lb in enumerate(LB):
                    if boolean and not boolize(lb): 
                        continue
                    C = CI[a, b]
                    if C >= cutoff:
                        na = (i, la)
                        nb = (j, lb)

                        if (na, nb) in G.edges():
                            C0 = G[na][nb]['weight']
                            if C > C0:
                                G.add_edge(nb, na, weight=C)
                                G.remove_edge(na, nb)
                        else:
                            G.add_edge(nb, na, weight=C)

                        G.nodes[nb]['index'] = j
                        G.nodes[nb]['label'] = lb

                        G.nodes[na]['index'] = i
                        G.nodes[na]['label'] = la

        # attach graph nodes to nodes in G
        X = np.arange(n_nodes)

        nodes = [node for node in G.nodes]

        for node in nodes:
            n, l = node
            x = X[L[n]==l]

            for i in x:
                ter = terminals[i]
                G.add_edge(node, ter, weight=1.)

        # remove grandparents (redundant edges)
        redundant = []
        for node in G.nodes():
            #parents = [_ for _ in G.predecessors(node)]
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

        G.remove_edges_from(redundant)

        # add a root node to the graph
        roots = []
        for node, indeg in G.in_degree():
            if indeg == 0:
                roots.append(node)

        if len(roots) > 1:
            root = (-1, 0)   # (-1, 0) will be changed to (0, 0) later
            G.add_node(root, index=-1, label=1)
            for node in roots:
                G.add_edge(root, node, weight=1.)

        self._full = G
        
        # find secondary edges
        secondary = []
        for node in G.nodes():
            parents = [_ for _ in G.predecessors(node)]
            if len(parents) > 1:
                nsize = self.node_size(node)
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
                    psize = self.node_size(p)
                    usize = w * nsize
                    j = usize / (nsize + psize - usize)
                    pref.append(j)

                # weight (CI) * node_size gives the size of the union between the node and the parent
                ranked_edges = [((x[0], node), x[1]) for x in sorted(zip(parents, pref), 
                                            key=lambda x: x[1], reverse=True)]
                secondary.extend(ranked_edges[1:])

        secondary.sort(key=lambda x: x[1], reverse=True)

        self._secondary = secondary

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

        self.hier = T

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

    def alldepths(self, leaf=True):
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

    def level_cluster(self, index):
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

        indices = nx.get_node_attributes(T, 'index')
        ancestors = [node for node in internals(T) if indices[node]==index]
                
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

def containment_indices1(A, B):
    LA = np.unique(A)
    LB = np.unique(B)

    CI = np.zeros((len(LA), len(LB)))
    for i, a in enumerate(LA):
        tfa = A == a
        na = np.sum(tfa)
        for j, b in enumerate(LB):
            tfb = B == b
            nb_in_a = np.sum(np.all([tfa, tfb], axis=0))
            CI[i, j] = float(nb_in_a)/na
    return CI, LA, LB

def containment_indices(A, B):
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

def containment_indices_legacy(A, B):
    if len(A) != len(B):
        raise ValueError('A and B must have the same length instead ' +\
                         'of (%d, %d)'%(len(A), len(B)))

    #A = np.asarray(A)
    #B = np.asarray(B)

    CA = Counter(A)
    CB = Counter(B)

    LA = [a for a in CA]
    LB = [b for b in CB]

    shape = (len(LA), len(LB))
    CI = np.zeros(shape)
    for i, a in enumerate(LA):
        #BinA = [B[i] for i in range(len(A)) if A[i]==a]
        BinA = B[A==a]
        counter = Counter(BinA)
        for b in counter:
            j = LB.index(b)
            CI[i, j] = counter[b] / float(CA[a])

    #LA = LA.tolist()
    #LB = LB.tolist()

    return CI, LA, LB

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
    assume_levels = kwargs.pop('assume_levels', False)
    boolean = kwargs.pop('boolean', False)

    weaver = Weaver(partitions, terminals, assume_levels=assume_levels, boolean=boolean)
    T = weaver.weave(**kwargs)

    return T

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

    w = Weaver(P, boolean=True, terminals=nodes, assume_levels=False)
    T = w.weave(cutoff=0.9, top=10)
