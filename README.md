# HiDeF (Hierarchical community Decoding Framework)
[![Documentation Status](https://readthedocs.org/projects/hidef/badge/?version=latest)](https://hidef.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/hidef/month)](https://pepy.tech/project/hidef)

<img src="https://github.com/fanzheng10/HiDeF/blob/master/fig1.png?raw=true" width="400">

## Introduction

HiDeF is a method for robustly resolving the hierarchical structures of networks based on multiscale community detection and the concepts of persistent homology. 

HiDeF is described in the following manuscript:  

Zheng, F., Zhang, S., Churas, C. et al., [HiDeF: identifying persistent structures in multiscale ‘omics data](https://doi.org/10.1186/s13059-020-02228-4). Genome Biol 22, 21 (2021).

## Updates

- `1.1.5` Fixed bug where ``_tmp`` edge list temp files collide if multiple instances of ``hidef_finder.py`` are run on same machine. 
          Made small fix to ``jaccard_matrix`` to handle scipy breaking [change](https://github.com/fanzheng10/HiDeF/commit/3dc6225cc67e59126b5b168996fb9718ea73d264)  
          
- `1.1.4` Add [Colab notebooks](https://github.com/fanzheng10/HiDeF/blob/master/analysis/protein_interaction_network_app.ipynb) allowing quick exploration of HiDeF results - now applicable to models based on protein-protein interaction network.  
- `1.1.3` Stable release around the time of paper publication, the first version available with `pip`.  

## Installation (Python package)

With pip:  
`pip install hidef`

From source:  
`python setup.py install`

## Usage

### Running HiDeF from Cytoscape

Best for small/medium networks < 10k nodes and < 50k edges.

HiDeF has been fully integrated with the [Cytoscape](https://cytoscape.org/) platform, via our recently published [Community Detection APplication and Service (CDAPS)](https://doi.org/10.1371/journal.pcbi.1008239) framework.

With this option users can access unique features in the CDAPS framework, including (1) interacting with the source network to visualize the subnetwork of any detected community (2) performing gene set enrichment analysis (when the vertices of the source network are proteins/genes) (3) sharing the models via the [NDEx](http://www.ndexbio.org/) database.

### Running HiDeF as a command-line tool

First, install the package as instructed above.

Using the codes in this repository, HiDeF can be used as a command-line tool. There are two main components of the scripts: `hidef_finder.py` and `weaver.py`.

To sweep the resolution profile and generate an optimized hierarchy based on pan-resolution community persistence, run the following command in a terminal: 

`python hidef_finder.py --g $graph --maxres $n --o $out [--options]`

- `$graph`: a tab delimited file with 2-3 columns: nodeA, nodeB, weight (optional).
- `$maxres`: the upper limit of the sampled range of the resolution parameter.
- `$out`: a prefix string for the output files.  

Other auxiliary parameters are explained in the manuscript.


#### Outputs
- `$out.nodes`: A TSV file describing the content (nodes in the input network) of each community. The last column of this file contains the persistence of each community.  
- `$out.edges`: A TSV file describing the parent-child relationships of communities in the hierarchy. The parent communities are in the 1st column and the children communities are in the 2nd column.  
- `$out.gml`: A file in the GML format that can be opened in Cytoscape to visualize the hierarchy (using "yFiles hierarchic layout" in Cytoscape)


### Using HiDeF as a python package

For documents, please see [https://hidef.readthedocs.io](https://hidef.readthedocs.io).

The following example shows how to build a hierarchical view of a network based on pre-computed communities, by using HiDeF as a Python package. This workflow only involves `weaver.py`.

First, the user needs to provide the clustering results on these data points. These results may be obtained from any multilevel clustering algorithm of the user's choice. In this example, suppose we have 8 data points and define 7 ways of partitioning them (in a Python terminal), 

```
P = ['11111111',
  '11111100',
  '00001111',
  '11100000',
  '00110000',
  '00001100',
  '00000011']
```

Then the hierarchical view can be obtained by

```
from hidef import weaver
wv = weaver.Weaver()
H = wv.weave(P, cutoff=1.0)
```
