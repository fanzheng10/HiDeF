from hidef import weaver

import pandas as pd

def hier_digraph_to_dash(H, coordinates=None, hide_terminal_nodes=False, add_node_data=False):
  '''
  Convert a digraph or weaver object (created in HiDeF) to a format readable by dash-cytoscape.

  Parameters
  ----------
  H: nx.classes.digraph.DiGraph or weaver.Weaver
      contain the information of the hierarchy
  coordinates: dict
      a dictionary mapping node names and node coordinates
  hide_terminal_nodes: boolean
      if set to True, do not draw terminal nodes (only applies when input is a weaver object)
  
  Returns
  ----------
  elements: a list of dictionaries; each element is either a node or an edge

  '''
  # be careful with "hide_terminal_nodes" option: as the input could take two forms
  elements = []
  element_index = {}
  isWeaver = False
  if isinstance(H, weaver.Weaver):
    H = H.hier
    isWeaver = True

  # add all the nodes
  for v in H.nodes():
    if isinstance(v, tuple):
      node_name = '-'.join(map(str, v))
    else:
      node_name = str(v)
      if isWeaver and hide_terminal_nodes:
        continue
    elements.append({'data': {'id': node_name, 'label': node_name.replace('Cluster', '')}})
    element_index[node_name] = len(elements) - 1

    if add_node_data:
      node_data_dict = H.nodes[v]
      _ = node_data_dict.pop('id', None)
      _ = node_data_dict.pop('label', None)
      elements[-1]['data'].update(node_data_dict)

  # add all the edges
  for e in H.edges():
    if isinstance(e[0], tuple):
      source_name = '-'.join(map(str, e[0]))
    else:
      source_name = str(e[0])
    if isinstance(e[1], tuple):
      target_name = '-'.join(map(str, e[1]))
    else:
      target_name = str(e[1])
      if isWeaver and hide_terminal_nodes:
        continue
    elements.append({'data': {'source': source_name, 'target': target_name}})
    edge_name = source_name + ',' + target_name
    element_index[edge_name] = len(elements) - 1

  if coordinates != None:
    assert isinstance(coordinates, dict), 'coordinates must be a dictionary!'
    for v, coord in coordinates.items():
      if isinstance(v, tuple):
        node_name = '-'.join(map(str, v))
      else:
        node_name = str(v)
      if not node_name in element_index:
        continue
      elements[element_index[node_name]]['position'] = {'x':coord[0], 'y':coord[1]}

  return elements

def subnet_to_dash(df_net, node_names):
  '''
  Create a subgraph (for dash-cytoscape)

  Parameters
  ----------
  df_net: pandas.DataFrame
      a dataframe representing a graph
  node_names: list
      a list of node names used to filter the graph to get a subgraph

  Returns
  ----------
  elements: a list of dictionaries; each element is either a node or an edge
  '''
  df_net_sub = df_net.loc[df_net[0].isin(node_names) & df_net[1].isin(node_names), :]
  elements = []
  node_dict = {}
  for _, row in df_net_sub.iterrows():
    v, w = row[0], row[1]
    if not v in node_dict:
      node_dict[v] = True
      elements.append({'data': {'id': v, 'label': v}})
    if not w in node_dict:
      node_dict[w] = True
      elements.append({'data': {'id': w, 'label': w}})
    elements.append({'data': {'source': v, 'target': w}})
  return elements


def get_hier_specs(H):
  names = []
  persistence_dict = {}
  size_dict = {}
  for k, v in H.nodes(data=True):
    names.append(k)
    if 'Stability' in v.keys():
      persistence_dict[k] = v['Stability']
    if 'Size' in v.keys():
      size_dict[k] = v['Size']
  df = pd.DataFrame(index=names)
  df['Persistence'] = pd.Series(persistence_dict)
  df['Size'] = pd.Series(size_dict)
  return df
