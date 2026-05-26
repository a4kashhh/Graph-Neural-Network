# PyTorch Geometric Graph Neural Networks

## Node Classification, Graph Classification & Scalable GNN Training using PyTorch Geometric

---

# Overview

This repository contains implementations and experiments on Graph Neural Networks (GNNs) using PyTorch and PyTorch Geometric.

The project covers:

- Graph Convolutional Networks (GCNs)
- Node Classification
- Graph Classification
- Mini-Batch GNN Training
- Cluster-GCN for Large Graphs
- Message Passing
- Graph Embedding Generation
- Pooling Mechanisms
- Embedding Visualization

Datasets used:

- Cora
- PubMed
- MUTAG

---

# Table of Contents

1. Introduction
2. Graph Neural Networks
3. Mathematical Foundations
4. Datasets
5. Node Classification
6. Graph Classification
7. Cluster-GCN
8. Training Pipeline
9. Installation
10. Usage
11. References

---

# Introduction

Graph Neural Networks (GNNs) are deep learning models specifically designed for graph-structured data.

Unlike traditional neural networks that operate on:
- images
- text
- sequences

GNNs operate on:
- nodes
- edges
- graph structures

Applications include:
- social networks
- citation networks
- molecular graphs
- recommendation systems
- biological networks

---

# Graph Neural Networks

A graph is represented as:

```math
G = (V,E)
```

Where:

- \(V\) = set of nodes
- \(E\) = set of edges

Each node contains a feature vector:

```math
x_v \in \mathbb{R}^d
```

Where:
- \(x_v\) is node feature vector
- \(d\) is feature dimension

---

# Graph Convolution

Graph Convolutional Networks update node representations using neighborhood information.

GCN propagation rule:

```math
H^{(l+1)} =
\sigma
\left(
\hat{D}^{-1/2}
\hat{A}
\hat{D}^{-1/2}
H^{(l)}
W^{(l)}
\right)
```

Where:

- \(\hat{A} = A + I\) is adjacency matrix with self-loops
- \(\hat{D}\) is degree matrix
- \(W^{(l)}\) are learnable weights
- \(\sigma\) is activation function

---

# Message Passing

Nodes aggregate information from neighbors:

```math
h_v^{(k+1)} =
\text{UPDATE}^{(k)}
\left(
h_v^{(k)},
\sum_{u \in N(v)}
\text{MESSAGE}^{(k)}
(h_v^{(k)}, h_u^{(k)})
\right)
```

Where:
- \(N(v)\) denotes neighbors of node \(v\)

---

# Activation Function

ReLU activation:

```math
f(x)=\max(0,x)
```

Used to introduce non-linearity.

---

# Dropout

Dropout randomly disables neurons during training:

```math
\text{Dropout}(x)=x \cdot m
```

Where:
- \(m\) is random binary mask

Purpose:
- reduce overfitting
- improve generalization

---

# Datasets

## 1. Cora

Citation network dataset.

### Properties
- Nodes: Research Papers
- Edges: Citations
- Task: Node Classification

---

## 2. PubMed

Biomedical citation graph.

### Properties
- Nodes: Scientific Papers
- Edges: Citations
- Task: Node Classification

---

## 3. MUTAG

Molecular graph dataset.

### Properties
- Nodes: Atoms
- Edges: Chemical Bonds
- Task: Graph Classification

---

# Node Classification using GCN

Node classification predicts labels for individual nodes.

Pipeline:

```text
Node Features
↓
GCN Layer
↓
Neighbor Aggregation
↓
ReLU
↓
Dropout
↓
GCN Layer
↓
Class Scores
```

---

# Node Classification Model

```python
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(
            dataset.num_node_features,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            dataset.num_classes
        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = x.relu()

        x = F.dropout(
            x,
            p=0.5,
            training=self.training
        )

        x = self.conv2(x, edge_index)

        return x
```

---

# Graph Classification using GCN

Graph classification predicts labels for entire graphs.

Pipeline:

```text
Node Features
↓
GCN Layers
↓
Node Embeddings
↓
Global Mean Pooling
↓
Graph Embedding
↓
Linear Classifier
↓
Graph Prediction
```

---

# Pooling Layer

Global Mean Pooling:

```math
h_G =
\frac{1}{|V|}
\sum_{v \in V}
h_v
```

Converts:
- many node embeddings
- into one graph embedding

---

# Graph Classification Model

```python
class GCN(torch.nn.Module):

    def __init__(self, hidden_channels):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(
            dataset.num_node_features,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.conv3 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.lin = Linear(
            hidden_channels,
            dataset.num_classes
        )

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(
            x,
            p=0.5,
            training=self.training
        )

        x = self.lin(x)

        return x
```

---

# Cluster-GCN for Large Graphs

Large graphs cannot always fit into GPU memory.

Cluster-GCN solves this by:
- partitioning large graphs
- training on graph clusters

---

# Graph Partitioning

```python
cluster_data = ClusterData(
    data,
    num_parts=128
)
```

This divides:
- one large graph
- into 128 subgraphs

---

# Cluster Loader

```python
train_loader = ClusterLoader(
    cluster_data,
    batch_size=32,
    shuffle=True
)
```

This creates:
- mini-batches of graph partitions

---

# Training Pipeline

Training workflow:

```text
Forward Pass
↓
Loss Computation
↓
Backpropagation
↓
Gradient Update
↓
Repeat
```

---

# Cross Entropy Loss

Used for classification tasks:

```math
L = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)
```

Where:
- \(y_i\) = true label
- \(\hat{y}_i\) = predicted probability

---

# Accuracy

Accuracy metric:

```math
Accuracy =
\frac{
\text{Correct Predictions}
}{
\text{Total Predictions}
}
```

---

# Embedding Visualization

t-SNE is used for visualizing learned embeddings in 2D space.

Pipeline:

```text
High-Dimensional Embeddings
↓
t-SNE
↓
2D Projection
↓
Visualization
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
```

---

# Install Dependencies

```bash
pip install torch
pip install torch-geometric
pip install matplotlib
pip install scikit-learn
```

---

# Usage

## Run Node Classification

```bash
python node_classification.py
```

---

## Run Graph Classification

```bash
python graph_classification.py
```

---

## Run Cluster-GCN

```bash
python cluster_gcn.py
```

---

# Technologies Used

- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- Scikit-Learn

---

# Learning Outcomes

This project demonstrates:

- Graph Representation Learning
- Message Passing in GNNs
- Node Embedding Generation
- Graph Embedding Generation
- Mini-Batch GNN Training
- Scalable Graph Learning
- Pooling Mechanisms
- Graph Classification
- Node Classification

---

# References

## Papers

1. Semi-Supervised Classification with Graph Convolutional Networks  
2. FastGCN  
3. Cluster-GCN  
4. Graph Neural Networks: A Review of Methods and Applications

---

# License

This project is open-source and available under the MIT License.
