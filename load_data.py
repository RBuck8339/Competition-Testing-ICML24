import networkx as nx
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import csv
from torch_geometric.data import Data

class loader():
    def __init__(self):
        self.my_graph = nx.DiGraph()
        self.alpha = 9
        self.folder_label_mapping = {
            'assets': 0,
            'derivatives': 1,
            'dex': 2,
            'lending': 3
        }
    
    
    def to_torch_obj(self):
        edges = []
        weights = []
        timestamps = []
        for u, v, data in G.edges(data=True):
            edges.append((u, v))
            weights.append(data['weight'])
            timestamps.append(data['timestamp'])
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float)
        edge_timestamp = torch.tensor(timestamps, dtype=torch.float)
        data = Data(edge_index=edge_index, edge_weight=edge_weight, edge_timestamp=edge_timestamp)
        torch.save(data, 'data.pt')
    
    def formula(self, curr, max, min):
        return 1 / 1 + (self.alpha * ((curr - min) / (max - min))))
    
    
    def hex_to_int(hex_id):
        return int(hex_id, 16)
    
    
    def visualize(self):
        pos = nx.spring_layout(self.my_graph)
        nx.draw(self.my_graph, pos, node_size=10, font_size=10, arrows=True)
        edge_labels = {(u, v): round(d['calculated_val'], 2) for u, v, d in self.my_graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.my_graph, pos,edge_labels=edge_labels, alpha=0.6, font_color='red', font_size=8)
        plt.show()
    
    
    def get_mapping(self, folder) -> dict:
        id_mapping = {}  # Dictionary to store the mapping from integer ID to (string ID, label)
        
        for root, dirs, files in os.walk(folder):
            folder_name = os.path.basename(root)
            if folder_name in self.folder_label_mapping:
                label = self.folder_label_mapping[folder_name]
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        with open(file_path, mode='r') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                hex_id = row['HexID']  # Adjust column name as needed
                                string_id = row['StringID']  # Adjust column name as needed
                                int_id = self.hex_to_int(hex_id)
                                id_mapping[int_id] = (string_id, label)
        return id_mapping


    def label_nodes(self):
        for node in self.my_graph.nodes():
            if node in self.id_mapping:
                string_id, label = self.id_mapping[node]
                self.my_graph.nodes[node]['string_id'] = string_id
                self.my_graph.nodes[node]['label'] = label
    
    
    def read_graph(self):
        data = pd.read_csv('networkaragonTX.txt', sep=' ', header=None)
        data.columns = ['From', 'To', 'UnixTime', 'Weight']

        data['From'] = data['From'].astype(int)
        data['To'] = data['To'].astype(int)
        data['Weight'] = data['Weight'].apply(int)

        data['Date'] = pd.to_datetime(data['UnixTime'], unit='s')
        data['Date'] = data['Date'].dt.strftime("%Y-%m-%d")

        data.sort_values(by='Date', inplace=True)
        
        dates = np.array(list(data['Date'].unique()))
        date = dates.min()
        curr_df = data[data['Date'] == date]  # Get the date into Year-Month-Date format (NO TIME) and only the rows with this
        max = curr_df['Weight'].max()
        min = curr_df['Weight'].min()

        # curr_key = date  # For many graphs

        # my_graphs[curr_key] = nx.DiGraph()  # Establish a new directed graph

        for row_idx in range(curr_df.shape[0]):
            curr_row = curr_df.iloc[row_idx]
            calculated = self.formula(curr_row['Weight'], max, min)

            # Add in both directions
            self.my_graph.add_edge(curr_row['From'], curr_row['To'], timestamp=int(curr_row['UnixTime']), weight=float(curr_row['Weight']), calculated_val = calculated)
            #my_graphs[curr_key].add_edge(curr_row['To'], curr_row['From'], timestamp=int(curr_row['UnixTime']), weight=float(curr_row['Weight']), calculated_val = calculated)