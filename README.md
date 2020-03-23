# Graph embedding convolutional network

Demonstration of implementing a fully parallelisable message passing graph convolutional network.
The Python files within this repository are Jupyter notebooks that can be opened with the [Jupytext extension](https://github.com/mwouts/jupytext).

Note: Turns out this has already been invented and is called a [Graph Attention Network](https://arxiv.org/abs/1710.10903) so take a squiz at that paper for the real deal.

## Creating training data

The notebook is designed to created training data from graph benchmark datasets downloaded from the TU Dortmund [Benchmark Data Sets for Graph Kernels](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

## Concept

![alternative textimus](./GECN.png?raw=true "Single GECN node update")
