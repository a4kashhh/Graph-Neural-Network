# GraphicalNeuralNetwork

A beginner-friendly implementation and explanation of Graph Neural Networks (GNNs) using PyTorch Geometric (PyG).

This repository demonstrates the fundamentals of Graph Neural Networks, Message Passing, Node Classification, and Graph Representation Learning through practical implementations on real-world graph datasets such as the Cora Citation Network and Zachary’s Karate Club Network.

---

# Project Overview

Graph Neural Networks (GNNs) are Deep Learning models specifically designed for graph structured data.

Unlike traditional neural networks, GNNs are capable of learning not only from node features, but also from the relationships and connections between nodes.

This project explores:
- Graph Neural Networks (GNNs)
- Message Passing
- Aggregation and Update Functions
- Node Embeddings
- Node Classification
- Graph Visualization
- Multi Layer Perceptron (MLP)
- Graph Convolutional Networks (GCN)
- Embedding Visualization using t-SNE
- PyTorch Geometric (PyG) Fundamentals

---

# What are Graph Neural Networks?

Graph Neural Networks are neural networks designed for graph data structures.

A graph mainly consists of:
- Nodes → objects/entities
- Edges → relationships/connections between nodes

Examples of graph data:
- Social Networks
- Citation Networks
- Recommendation Systems
- Molecular Structures
- Web Networks
- Road Networks

Traditional neural networks process data independently, while GNNs allow nodes to learn from their neighboring connected nodes.

---

# Message Passing in GNN

The core idea behind Graph Neural Networks is **Message Passing**.

In Message Passing:
- each node gathers information from neighboring nodes
- neighboring information is aggregated
- node representation is updated iteratively

General Message Passing Formula:

$$
h_v^{(k)} = UPDATE^{(k)} \left( h_v^{(k-1)}, AGGREGATE^{(k)} \left( \{ h_u^{(k-1)} : u \in N(v) \} \right) \right)
$$

Where:
- AGGREGATE → collects neighbor information
- UPDATE → updates node representation
- $\mathcal{N}(v)$ → neighboring nodes of node $v$

---

# Dataset Used

## 1. Zachary's Karate Club Network

A classical social network dataset used for community detection.

Features:
- 34 nodes
- Social relationship graph
- Community detection problem

---

## 2. Cora Citation Network

A benchmark graph dataset used for Node Classification.

In the Cora dataset:
- Nodes represent research papers
- Edges represent citation links
- Features represent important words inside documents

Dataset Statistics:
- 2708 Nodes
- 5429 Edges
- 1433 Input Features
- 7 Output Classes

---

# Multi Layer Perceptron (MLP)

Initially, a Multi Layer Perceptron (MLP) model was implemented for node classification.

Architecture:
- Linear Layer
- ReLU Activation
- Dropout
- Linear Output Layer

Workflow:

```text
Input Features
      ↓
Linear Layer
      ↓
ReLU Activation
      ↓
Dropout
      ↓
Linear Layer
      ↓
Output Classes
```

---

# Limitation of MLP

Although the MLP model was able to learn node features, it performed poorly on graph structured data.

Main limitations:
- treats each node independently
- ignores graph structure
- does not use neighboring node information
- suffers from overfitting
- weak node embeddings

The MLP model achieved approximately:
- ~59% Test Accuracy

Visualization showed:
- scattered embeddings
- weak class separation
- mixed node clusters

---

# Graph Convolutional Network (GCN)

To overcome the limitations of MLP, a Graph Convolutional Network (GCN) was implemented.

Unlike MLP, GCN learns using:
- node features
- neighboring node information
- graph connectivity structure

Architecture:
- GCNConv Layer
- ReLU Activation
- Dropout
- GCNConv Output Layer

Workflow:

```text
Node Features + Edge Connections
              ↓
         GCNConv
              ↓
       ReLU Activation
              ↓
           Dropout
              ↓
         GCNConv
              ↓
        Output Classes
```

---

# Why GCN Performs Better

GCN uses neighborhood aggregation through Message Passing.

Each node:
- collects information from neighboring nodes
- aggregates neighbor features
- updates its representation

This allows the model to learn:
- graph structure
- semantic relationships
- neighborhood patterns
- meaningful embeddings

---

# Performance Improvement using GCN

The transition from MLP to GCN resulted in significant performance improvement.

Improvements observed:
- better node embeddings
- stronger class separation
- improved generalization
- reduced overfitting
- more meaningful graph representations

Visualization after training clearly showed:
- compact node clusters
- better embedding organization
- improved class boundaries

This demonstrates the effectiveness of Graph Neural Networks for graph based learning tasks.

---

# Embedding Visualization

t-SNE visualization was used to project high-dimensional node embeddings into 2D space.

Visualization helps analyze:
- cluster formation
- node similarity
- embedding quality
- class separation

---

# Technologies and Libraries Used

- Python
- PyTorch
- PyTorch Geometric (PyG)
- NetworkX
- Matplotlib
- Scikit-learn

---

# Installation

```bash
pip install torch
pip install torch-geometric
pip install networkx
pip install matplotlib
pip install scikit-learn
```

---

# Project Goals

The main objectives of this repository are:
- Understanding Graph Neural Networks
- Learning Message Passing
- Implementing Node Classification
- Understanding Graph Embeddings
- Learning Graph Representation Learning
- Exploring PyTorch Geometric
- Comparing MLP and GCN architectures

---

# Real World Applications of GNNs

Graph Neural Networks are widely used in:
- Social Network Analysis
- Recommendation Systems
- Fraud Detection
- Citation Networks
- Knowledge Graphs
- Molecular Chemistry
- Traffic Prediction
- Protein Interaction Networks

---

# Future Improvements

Possible future extensions:
- Graph Attention Networks (GAT)
- GraphSAGE
- Heterogeneous Graph Learning
- Dynamic Graphs
- Link Prediction
- Graph Level Classification

---

# Conclusion

This project demonstrates how Graph Neural Networks outperform traditional neural networks on graph structured data.

By incorporating neighborhood information and graph connectivity, GCN models generate more meaningful node embeddings and achieve significantly better node classification performance compared to standard MLP models.

The repository serves as a beginner-friendly introduction to Graph Neural Networks and Graph Representation Learning using PyTorch Geometric.

---

# References

- PyTorch Geometric Documentation
- Kipf & Welling (2017) — Semi-Supervised Classification with Graph Convolutional Networks
- Cora Citation Network Dataset
- Zachary Karate Club Dataset
