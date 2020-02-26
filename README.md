# HiDef (Hierarchical community Decoding Framework)


## Introduction

A package for resolving the hierarchical structures of networks based on multiscale community detection. 

Corresponding to a paper submitted to ISMB 2020:  
*Robust and flexible decoding of multiscale structures in complex network data*, S Zhang*, F Zheng*, I Bahar, T Ideker.
(\* equal contribution)**

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

`python finder.py --g $graph --n $n [--options]`

## Inputs

`$graph`: a tab delimited file with 2-3 columns: nodeA, nodeB, weight (optional)  
`$n`: the upper limit of the sampled range of the resolution parameter 

## Outputs






