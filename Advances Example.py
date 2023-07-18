# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:11:55 2023

@author: muhamzs
Graph Convlution Autoencoder:
Example File for Uni-directed Graphs (Link Predicttion is also included)
"""

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.fc = Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.fc = Linear(out_channels, 1024)

    def forward(self, z, edge_index):
        z = F.relu(self.conv1(z, edge_index))
        z = self.conv2(z, edge_index)
        z = self.fc(z)
        return z


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.fc = torch.nn.Linear(2 * in_channels, 1)  # We're going to concatenate two feature vectors

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        x = self.fc(x)
        return torch.sigmoid(x)


class GCAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCAE, self).__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.decoder = Decoder(out_channels, in_channels)
        self.link_predictor = LinkPredictor(out_channels)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z, edge_index)
        edge_probs = self.link_predictor(z[edge_index[0]], z[edge_index[1]])
        return x_hat, edge_probs


num_nodes = 10
time_series_length = 1024
num_events = 1000
data_list = []

for _ in range(num_events):
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()
    x = torch.randn((num_nodes, time_series_length), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data_list.append(data)

loader = DataLoader(data_list, batch_size=32)

# Instantiate the autoencoder and optimizer
model = GCAE(time_series_length, 512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define the loss function
criterion = torch.nn.MSELoss()
bce_loss = torch.nn.BCELoss()

losses = []

# Training loop
for epoch in range(1):  
    for batch in loader:
        optimizer.zero_grad()
        x_hat, edge_probs = model(batch.x, batch.edge_index)
        loss = criterion(x_hat, batch.x) + bce_loss(edge_probs, torch.ones(edge_probs.shape).to(batch.x.device))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')
    losses.append(loss.item())


# Define a function to plot adjacency matrices
def plot_adjacency_matrix(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    for edge in edge_index.T:
        adj_matrix[edge[0], edge[1]] = 1
    plt.imshow(adj_matrix)
    plt.show()

# Define a function to plot node features
def plot_node_features(features):
    num_nodes, num_features = features.shape
    for node in range(num_nodes):
        plt.plot(features[node], label=f'Node {node}')
    plt.legend()
    plt.show()

# After training, plot the original and reconstructed adjacency matrix and node features
with torch.no_grad():
    event_idx = 0  # Select the event
    event = data_list[event_idx]
    x_hat, edge_probs = model(event.x, event.edge_index)

    # Plot the original and reconstructed adjacency matrix
    print("Original Adjacency Matrix:")
    plot_adjacency_matrix(event.edge_index, num_nodes)

    print("Reconstructed Adjacency Matrix:")
    reconstructed_edge_index = (edge_probs > 0.5).nonzero(as_tuple=False).T  # Threshold can be adjusted
    plot_adjacency_matrix(reconstructed_edge_index, num_nodes)

    # Plot the original and reconstructed node features
    print("Original Node Features:")
    plot_node_features(event.x)

    print("Reconstructed Node Features:")
    plot_node_features(x_hat)
