# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Load, inspect, and save training data
#
# This is for data from the [Benchmark Data Sets for Graph Kernels](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) page.
#
# What we need out is the adjacency matrix and feature matrix of each graph. 
#
# ## File Format
#
# The data sets have the following format (replace DS by the name of the data set):
#
# Let
#
#   * n = total number of nodes
#   * m = total number of edges
#   * N = number of graphs
#
# The files are:
#
#   1. DS_A.txt (m lines): sparse (block diagonal) adjacency matrix for all graphs, each line corresponds to (row, col) resp. (node_id, node_id). All graphs are undirected. Hence, DS_A.txt contains two entries for each edge.
#   2. DS_graph_indicator.txt (n lines): column vector of graph identifiers for all nodes of all graphs, the value in the i-th line is the graph_id of the node with node_id i
#   3. DS_graph_labels.txt (N lines): class labels for all graphs in the data set, the value in the i-th line is the class label of the graph with graph_id i
#   4. DS_node_labels.txt (n lines): column vector of node labels, the value in the i-th line corresponds to the node with node_id i
#
# There are optional files if the respective information is available:
#
#   5. DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt): labels for the edges in DS_A_sparse.txt
#   6. DS_edge_attributes.txt (m lines; same size as DS_A.txt): attributes for the edges in DS_A.txt
#   7. DS_node_attributes.txt (n lines): matrix of node attributes, the comma seperated values in the i-th line is the attribute vector of the node with node_id i
#   8. DS_graph_attributes.txt (N lines): regression values for all graphs in the data set, the value in the i-th line is the attribute of the graph with graph_id i*

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

dataset = 'PROTEINS_full'
top_dir = Path('/local/scratch/ssd2/jkahn/benchmarks/')
dataset_dir = top_dir / dataset

# ## Load the data

# +
# Load the graph components
indicator_df = pd.read_csv(dataset_dir / '{}_graph_indicator.txt'.format(dataset), header=None)
edges_df = pd.read_csv(dataset_dir / '{}_A.txt'.format(dataset), header=None)
labels_df = pd.read_csv(dataset_dir / '{}_graph_labels.txt'.format(dataset), header=None)
node_att_df = pd.read_csv(dataset_dir / '{}_node_attributes.txt'.format(dataset), header=None)

# And set names
indicator_df.columns = ['graph_num']
node_att_names = ['att_{}'.format(i) for i in range(len(node_att_df.columns))]
node_att_df.columns = node_att_names
# -

node_df = pd.concat([indicator_df, node_att_df], axis=1)

# Magic from https://stackoverflow.com/questions/52621497/pandas-group-by-column-and-transform-the-data-to-numpy-array
g = node_df.groupby('graph_num').cumcount()
feat_X = (
    node_df.set_index(['graph_num',g])
    .unstack(fill_value=0)
    .stack().groupby(level=0)
    .apply(lambda x: x.values.tolist())
    .tolist()
)
feat_X = np.array(feat_X)

feat_X.shape

graph_sizes = node_df.groupby('graph_num').size()
graph_sizes = graph_sizes.cumsum()

# ### Build the adjacency matrix
#
# Since they didn't give us a graph indicator for the edges I have to build this shit myself

indicator_df.shape

block_adj = lil_matrix((indicator_df.shape[0], indicator_df.shape[0]), dtype=np.int8)

edge_pairs = edges_df.values - 1

for pair in edge_pairs:
    block_adj[pair[0], pair[1]] = 1

adj_X = np.zeros((*feat_X.shape[:2], feat_X.shape[1]))

# Populate the adjacency matrix from the block sparse matrix
prev = 0
for i, size in enumerate(graph_sizes):
    sub_arr = block_adj[prev:size, prev:size].toarray()
    adj_X[i][0:(size-prev), 0:(size-prev)] = sub_arr
    prev=size

# ## Plot for sanity check

plt.figure()
plt.imshow(adj_X[0, :47, :47])

# ## Preprocess the features

# +
f, ax = plt.subplots(8, 4, figsize=(15, 15))
for idx in range(feat_X.shape[-1]):
#     print(idx, idx//4, idx%4)
    ax[int(idx//4), idx%4].hist(feat_X[:, :2, idx], bins=60)
    ax[int(idx//4), idx%4].title.set_text(idx)

# plt.figure()
# plt.hist(feat_X[:, 1, 28], bins='auto')
plt.show()
# -

plt.figure()
plt.hist(feat_X[:, 1, 4], bins='auto')
plt.show()

# +
# Do standard normalisation on all features (can do 4, will still need to one-hot encode that after)
x_min = feat_X.min(axis=(0, 1), keepdims=False)
x_max = feat_X.max(axis=(0, 1), keepdims=False)

# x_min.shape
norm_X = (feat_X - x_min)/(x_max-x_min)
norm_X.shape
# -

np.unique(norm_X[0, :, 4])

# Manually adding number of classes here, and the step size
cat_X = norm_X[:, :, 4]
one_hot = (np.arange(3) == (2*cat_X[...,None])).astype(int)
one_hot[:, :, 1].max()

# Delete the old column and insert one-hots
prep_X = np.delete(norm_X, 4, -1)
prep_X = np.concatenate([prep_X, one_hot], axis=-1)
prep_X.shape

# +
# Plot after prepalisation
f, ax = plt.subplots(8, 4, figsize=(15, 15))
for idx in range(prep_X.shape[-1]):
#     print(idx, idx//4, idx%4)
    ax[int(idx//4), idx%4].hist(prep_X[:, :2, idx], bins=60)
    ax[int(idx//4), idx%4].title.set_text(idx)

# plt.figure()
# plt.hist(feat_X[:, 1, 28], bins='auto')
plt.show()
# -

# ## Save the data

np.save(dataset_dir / 'feat_X.npy', feat_X)
np.save(dataset_dir / 'adj_X.npy', adj_X)
np.save(dataset_dir / 'y.npy', labels_df.values)
