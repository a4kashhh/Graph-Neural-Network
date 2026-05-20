# GraphicalNeuralNetwork

Introduction to Graph Neural Networks (GNNs) using PyTorch Geometric (PyG).

This repository contains beginner-friendly implementations and explanations of:
- Graph Neural Networks (GNNs)
- Message Passing
- Aggregation and Update Functions
- Graph Visualization
- Node Embeddings
- PyTorch Geometric Basics

---

# What are Graph Neural Networks?

Graph Neural Networks (GNNs) are Deep Learning models specially designed for graph structured data.

A graph contains:
- Nodes -> objects/entities
- Edges -> connections between nodes

Examples:
- Social Networks
- Road Networks
- Recommendation Systems
- Molecules
- Web Networks

---

# Message Passing in GNN

In GNNs, each node learns information from its neighboring nodes.

General Message Passing Formula:

$$
\mathbf{h}_v^{(k)} =
\text{UPDATE}^{(k)}
\left(
\mathbf{h}_v^{(k-1)},
\text{AGGREGATE}^{(k)}
\left(
\left\{
\mathbf{h}_u^{(k-1)} :
u \in \mathcal{N}(v)
\right\}
\right)
\right)
$$

Where:
- AGGREGATE -> collects neighbor information
- UPDATE -> updates node representation
- \( \mathcal{N}(v) \) -> neighbors of node \(v\)

---

# Features

- Graph Visualization using NetworkX
- Embedding Visualization using Matplotlib
- PyTorch Geometric Implementations
- Beginner Friendly Code
- Hands-on GNN Learning

---

# Libraries Used

- Python
- PyTorch
- PyTorch Geometric (PyG)
- NetworkX
- Matplotlib

---

# Installation

```bash
pip install torch
pip install torch-geometric
pip install networkx
pip install matplotlib
```

---

# Example Dataset

This project uses the famous:

## Zachary's Karate Club Network

- 34 nodes
- Social network graph
- Community detection problem

---

# Project Goal

The goal of this repository is to understand:
- How GNNs work
- Message Passing
- Node Embeddings
- Graph Representation Learning
- Graph based Deep Learning

---

# Author

Akash Pandey

Computer Science Undergraduate  
VIT Bhopal University

---

# Future Improvements

- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Node Classification
- Link Prediction
- Graph Classification
- Real-world datasets
