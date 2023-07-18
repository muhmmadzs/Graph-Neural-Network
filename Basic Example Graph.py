# -*- coding: utf-8 -*-
"""
Created on Friday  

@author: muhamzs
Graph Convlution Autoencoder:
Example File for Uni-directed Graphs

"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)
        self.fc = Linear(out_channels, out_channels)  # Fully connected layer

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.fc(x)  # apply FCN to each node's features individually
        return x

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)
        self.fc = Linear(out_channels, 1024)  # Ensure the output matches the original input dimension

    def forward(self, z, edge_index):
        z = F.relu(self.conv1(z, edge_index))
        z = self.conv2(z, edge_index)
        z = self.fc(z)  # apply FCN to each node's features individually
        return z


class GCAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCAE, self).__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.decoder = Decoder(out_channels, in_channels)  # we swap in/out for the decoder

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z, edge_index)
        return x_hat


num_nodes = 10
time_series_length = 1024
num_events = 1000
data_list = []

for _ in range(num_events):
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()
    x = torch.randn((num_nodes, time_series_length), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data_list.append(data)

loader = DataLoader(data_list, batch_size=5)

# Instantiate the autoencoder and optimizer
model = GCAE(time_series_length, 512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
criterion = torch.nn.MSELoss()

losses = []

# Training loop
for epoch in range(20):  # Or however many epochs you want to train for
    for batch in loader:
        optimizer.zero_grad()
        x_hat = model(batch.x, batch.edge_index)
        loss = criterion(x_hat, batch.x)
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')
    losses.append(loss.item())

# Plot the training loss
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# After training, let's compare original and reconstructed time series for a node in an event
with torch.no_grad():
    event_idx = 0  # Select the event
    node_idx = 0  # Select the node
    event = data_list[event_idx]
    x_hat = model(event.x, event.edge_index)
    
    plt.figure()
    plt.plot(event.x[node_idx].numpy(), label='Original')
    plt.plot(x_hat[node_idx].numpy(), label='Reconstructed')
    plt.legend()
    plt.show()