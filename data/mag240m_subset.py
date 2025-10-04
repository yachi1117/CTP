import os
import torch
import numpy as np
from ogb.lsc import MAG240MDataset
from torch_geometric.data import Data
import scipy.sparse as sp
from tqdm import tqdm

def get_neighbors(adj, nodes):
    """Get both incoming and outgoing neighbors of nodes in a sparse matrix"""
    # Get outgoing neighbors (rows)
    out_neighbors = set()
    for node in nodes:
        out_neighbors.update(adj.indices[adj.indptr[node]:adj.indptr[node + 1]])
    
    # Get incoming neighbors (columns) using transpose
    adj_T = adj.T.tocsr()
    in_neighbors = set()
    for node in nodes:
        in_neighbors.update(adj_T.indices[adj_T.indptr[node]:adj_T.indptr[node + 1]])
    
    return out_neighbors.union(in_neighbors)

def create_mag_subset_efficient(root, subset_size=20000000, save_dir=None, chunk_size=1000000):
    """
    Create and save a connected subset of MAG240M with memory-efficient processing
    """
    if save_dir is None:
        save_dir = os.path.join(root, "mag240m_subset")
    os.makedirs(save_dir, exist_ok=True)
    
    subset_path = os.path.join(save_dir, f"mag240m_subset_{subset_size}.pt")
    print(subset_path)
    if os.path.exists(subset_path):
        print(f"Loading existing subset from {subset_path}")
        return torch.load(subset_path)
    
    print("Loading MAG240M dataset...")
    dataset = MAG240MDataset(root)
    num_nodes = dataset.num_papers
    
    print("Building sparse adjacency matrix...")
    edge_index = dataset.edge_index('paper', 'cites', 'paper')
    rows, cols = edge_index[0], edge_index[1]
    print(f"Total edges: {len(rows)}")
    
    # Create sparse adjacency matrix
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                       shape=(num_nodes, num_nodes))
    
    print("Sampling connected subgraph...")
    # Start from a well-connected node (choose one with many neighbors)
    degrees = np.array(adj.sum(axis=1)).flatten() + np.array(adj.sum(axis=0)).flatten()
    start_node = np.argmax(degrees)
    subset_nodes = set([start_node])
    frontier = list(subset_nodes)
    
    pbar = tqdm(total=subset_size)
    pbar.update(1)
    
    while len(subset_nodes) < subset_size and frontier:
        # Process frontier in chunks for memory efficiency
        chunk = frontier[:10000]
        frontier = frontier[10000:]
        
        # Get neighbors using sparse matrix operations
        neighbors = get_neighbors(adj, chunk)
        
        # Add new nodes to frontier
        new_nodes = neighbors - subset_nodes
        if new_nodes:
            subset_nodes.update(new_nodes)
            frontier.extend(new_nodes)
            pbar.update(len(new_nodes))
            
        if len(subset_nodes) >= subset_size:
            break
            
        # If frontier is empty but we haven't reached target size,
        # add a new random node from high-degree nodes not yet visited
        if not frontier and len(subset_nodes) < subset_size:
            remaining_nodes = set(range(num_nodes)) - subset_nodes
            remaining_degrees = degrees.copy()
            remaining_degrees[list(subset_nodes)] = 0
            new_start = np.argmax(remaining_degrees)
            frontier.append(new_start)
            subset_nodes.add(new_start)
            pbar.update(1)
    
    pbar.close()
    
    # Convert to list and tensor
    subset_nodes = torch.tensor(sorted(list(subset_nodes)))[:subset_size]
    print(f"\nCollected {len(subset_nodes)} nodes")
    
    print("Creating node mapping...")
    # Create node mapping
    node_idx = torch.zeros(num_nodes, dtype=torch.long)
    node_idx[subset_nodes] = torch.arange(len(subset_nodes))
    
    print("Getting edges for subset...")
    # Convert edges to tensor
    edge_index = torch.from_numpy(edge_index)
    # Find edges where both nodes are in the subset
    mask = torch.isin(edge_index[0], subset_nodes) & torch.isin(edge_index[1], subset_nodes)
    subset_edges = edge_index[:, mask]
    # Remap node indices
    subset_edges = node_idx[subset_edges]
    
    print("Getting node features...")
    # Get features in chunks to save memory
    subset_feats = []
    for i in tqdm(range(0, len(subset_nodes), chunk_size)):
        chunk_nodes = subset_nodes[i:i+chunk_size].numpy()
        feat_chunk = torch.from_numpy(dataset.paper_feat[chunk_nodes])
        subset_feats.append(feat_chunk)
    subset_feats = torch.cat(subset_feats, dim=0)
    
    # Create and save subset graph
    subset_graph = Data(
        x=subset_feats,
        edge_index=subset_edges,
        num_nodes=len(subset_nodes)
    )
    
    print(f"Saving subset to {subset_path}")
    torch.save(subset_graph, subset_path)
    torch.save(subset_nodes, os.path.join(save_dir, f"mag240m_subset_{subset_size}_nodes.pt"))
    
    print(f"Final graph stats:")
    print(f"Nodes: {subset_graph.num_nodes}")
    print(f"Edges: {subset_graph.edge_index.size(1)}")
    print(f"Average degree: {subset_graph.edge_index.size(1) / subset_graph.num_nodes:.2f}")
    
    return subset_graph

if __name__ == "__main__":
    ROOT = "DATA_ROOT/mag240m"  # Replace with your data root directory path
    subset_graph = create_mag_subset_efficient(ROOT, subset_size=20000000)