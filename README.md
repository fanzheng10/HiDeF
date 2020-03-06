# HiDeF (Hierarchical community Decoding Framework)


## Introduction

A package for resolving the hierarchical structures of networks based on multiscale community detection. 

Corresponding to a paper submitted to ISMB 2020:  
*Robust and flexible decoding of multiscale structures in complex network data*, S Zhang*, F Zheng*, I Bahar, T Ideker.
(\* equal contribution)

There are two main components of the scripts: `finder.py` and `weaver.py`

Detailed documentations, including notebooks describing HiDeF usages in analyzing biological datasets, will be provided soon.

## Dependencies (with tested versions)

[networkx](https://networkx.github.io/) 2.3  
[python-igraph](https://igraph.org/python/) 0.7.1  
[louvain-igraph](https://github.com/vtraag/louvain-igraph) 0.6.1  
[leidenalg](https://github.com/vtraag/leidenalg)    0.7.0  
numpy  
scipy  
pandas


## Usage

### Hierarchical view given pre-computed communities
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
>>> from hidef import weaver
>>> weaver = Weaver(P, boolean=True, terminals=nodes, assume_levels=False)
>>> T = weaver.weave(cutoff=0.9, top=10)
```

The hierarchy is represented by a `networkx.DiGraph` object, which can be obtained by querying `T.hier`. `T` also contains a lot of useful functions for extracting useful information about the hierarchy. 

### Pan-resolution community detection of a network

To sweep the resolution profile and generate an optimized hierarchy based on pan-resolution community persistence, run the following command in a terminal: 

`python finder.py --g $graph --n $n --o $out [--options]`

`$graph`: a tab delimited file with 2-3 columns: nodeA, nodeB, weight (optional).

`$maxres`: the upper limit of the sampled range of the resolution parameter.

`$out`: a prefix string for the output files.  

Other auxiliary parameters are explained in the supplemental material of the manuscript (to be provided soon).

#### Outputs
`$out.nodes`: A file describing the content (nodes in the input network) of each community.  
`$out.edges`: A file describing the parent-child relationships of communities in the hierarchy. The parent communities are in the 1st column and the children communities are in the 2nd column.  
`$out.gml`: A file in the GML format that can be opened in Cytoscape.
