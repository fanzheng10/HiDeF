# HiDeF (Hierarchical community Decoding Framework)


## Introduction

A package for resolving the hierarchical structures of networks based on multiscale community detection. 

Corresponding to a paper submitted to ISMB 2020:  
*Robust and flexible decoding of multiscale structures in complex network data*, S Zhang*, F Zheng*, I Bahar, T Ideker.
(\* equal contribution)

There are two main components of the scripts: `finder.py` and `weaver.py`

Detailed documentations are coming.

## Dependencies (with tested versions)

[networkx](https://networkx.github.io/) 2.3  
[python-igraph](https://igraph.org/python/) 0.7.1  
[louvain-igraph](https://github.com/vtraag/louvain-igraph) 0.6.1  
[leidenalg](https://github.com/vtraag/leidenalg)    0.7.0  
numpy  
scipy  
pandas


## Usage

# Hierarchical view
The following example shows how to obtain a hierarchical view of given data points using HiDeF. 

First, the user needs to provide the clustering results on these data points. These results may be obtained from any multilevel clustering algorithm of user's choice. In this example, suppose we have 8 data points and define 7 ways of partitioning them (in a Python terminal), 

```
>>> P = ['11111111',
...      '11111100',
...      '00001111',
...      '11100000',
...      '00110000',
...      '00001100',
...      '00000011']
```

The labels of these nodes are assigned as follows (optional):

```
>>> nodes = 'ABCDEFGH'
```

Then the hierarchical view can be obtained by

```
>>> weaver = Weaver(P, boolean=True, terminals=nodes, assume_levels=False)
>>> T = weaver.weave(cutoff=0.9, top=10)
```

The hierarchy is represented by a `networkx.DiGraph` object, which can be obtained by querying `T.hier`. `T` also contains a lot of useful functions for extracting useful information about the hierarchy. 

# Optimal resolution
To identify the optimal Louvain modularity of a graph, simply run the following command in a terminal: 

`python finder.py --g $graph --n $n [--options]`

`$graph`: a tab delimited file with 2-3 columns: nodeA, nodeB, weight (optional).

`$n`: the upper limit of the sampled range of the resolution parameter.

