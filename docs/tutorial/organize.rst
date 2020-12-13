Organize a list of clusters into a DAG
---------------------------------------

HiDeF is flexible and the construction of DAG is independent of the upstream clustering algorithm. The ``weaver`` module can take an arbitrary list of clusters and infer the relationships among them. We defined containment index

.. math:: CI(v,w) = \frac{|s(v) \cap s(w)|}{|s(w)|}

as the fraction of data points in cluster *w* that is also in cluster *v*. In the following example, all data points of cluster at line 2 is also members of the cluster at line 1, so we use a directed edge to represent this relationship.

.. code-block:: python

    from hidef import *
    P = ['11111111',
         '11111100',
         '00001111',
         '11100000',
         '00110000',
         '00001100',
         '00000011']
    wv = weaver.Weaver()
    H = wv.weave(P, cutoff=1.0)



In many cases, it could be informative to relax the requirement of :math:`CI = 1` to reveal the relationships between clusters. Thus we provide a parameter :math:`\tau` ("cutoff" in the above code block) to approximate the containment relationships, as can be seen in the following example. ::

   Code

.. Note::
    Since we do not restrict the input, it is possible to have mutually contained clusters. In such cases, we decide the two (or more) mutually contained clusters as “similar”, and merge them into a new cluster. In the DAG, the input and output edges of this new node are the union of the edges in and out of the nodes in the merged group.


Although an arbitrarily defined set of clusters can be used, a routine workflow is to use network community detection to generate a set of clusters (see the next topic).