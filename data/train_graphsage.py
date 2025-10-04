import os
import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import numpy as np
import gc

# For the MAG240M dataset
from ogb.lsc import MAG240MDataset
from torch_geometric.data import Data

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Apply ReLU for all but the last layer
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return x
    
    def get_embeddings(self, data_loader, device, original_node_indices, embedding_dim):
        """
        Extract embeddings for all nodes while preserving the mapping to original MAG240M node IDs
        
        Args:
            data_loader: NeighborLoader for the graph
            device: Device to run inference on
            original_node_indices: Tensor of original node indices from MAG240M
            embedding_dim: Dimension of the embeddings
        
        Returns:
            embeddings: Tensor of shape [num_subset_nodes, embedding_dim]
            id_mapping: Mapping dictionary from position in embeddings to original node ID
        """
        self.eval()
        
        # Create mapping from remapped indices (0 to subset_size-1) to original node indices
        remapped_to_original = {i: original_id.item() for i, original_id in enumerate(original_node_indices)}
        
        # Initialize embeddings tensor - using the SUBSET size (not original node space size)
        num_nodes = len(original_node_indices)
        all_embeddings = torch.zeros(num_nodes, embedding_dim, device='cpu')
        
        # Track processed nodes to ensure we don't miss any
        processed_nodes = set()
        
        # Process batches
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting embeddings"):
                batch = batch.to(device)
                batch_embeddings = self.forward(batch.x, batch.edge_index)
                
                # Get remapped indices for central nodes in the batch
                # These are the original positions in the subset graph (0 to subset_size-1)
                batch_indices = batch.n_id[:batch.batch_size].cpu().numpy()
                
                # Store embeddings at the EXACT same positions as in the subset
                for i, idx in enumerate(batch_indices):
                    all_embeddings[idx] = batch_embeddings[i].cpu()
                    processed_nodes.add(idx)
                
                del batch, batch_embeddings
                torch.cuda.empty_cache()
        
        # Verify that we have embeddings for all nodes in the subset
        expected_nodes = set(range(num_nodes))
        missing_nodes = expected_nodes - processed_nodes
        if missing_nodes:
            print(f"Warning: {len(missing_nodes)} nodes are missing embeddings")
        else:
            print(f"Successfully extracted embeddings for all {len(processed_nodes)} nodes")
            print(f"Embedding indices match exactly with the subset node indices (0 to {num_nodes-1})")
        
        return all_embeddings, remapped_to_original

def create_mag240m_subset(root_dir, subset_size=20000000, random_seed=42):
    """Create a subset of the MAG240M dataset"""
    # Define paths
    subset_dir = os.path.join(root_dir, 'mag240m_subset')
    os.makedirs(subset_dir, exist_ok=True)
    
    node_path = os.path.join(subset_dir, f'mag240m_subset_{subset_size}_nodes.pt')
    edge_path = os.path.join(subset_dir, f'mag240m_subset_{subset_size}_edges.pt')
    feature_path = os.path.join(subset_dir, f'mag240m_subset_{subset_size}_features.pt')
    graph_path = os.path.join(subset_dir, f'mag240m_subset_{subset_size}_graph.pt')
    
    # Check if the subset already exists
    if os.path.exists(graph_path):
        print(f"Loading existing MAG240M subset from {graph_path}")
        data = torch.load(graph_path)
        node_indices = torch.load(node_path)
        print(f"Loaded graph with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
        return data, node_indices
    
    # Otherwise, create the subset
    print(f"Creating MAG240M subset with {subset_size} nodes")
    
    # Load the full dataset
    dataset = MAG240MDataset(root=root_dir)
    
    # Sample nodes
    np.random.seed(random_seed)
    num_papers = dataset.num_papers
    
    # Sample paper nodes
    if subset_size < num_papers:
        node_indices = np.random.choice(num_papers, subset_size, replace=False)
        node_indices = np.sort(node_indices)
        node_indices = torch.tensor(node_indices)
    else:
        node_indices = torch.arange(num_papers)
    
    torch.save(node_indices, node_path)
    
    # Create node id mapping
    node_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(node_indices)}
    
    # Get paper features for the subset
    print("Loading features for subset nodes")
    features = torch.from_numpy(dataset.paper_feat[node_indices]).float()
    torch.save(features, feature_path)
    
    # Get edges
    print("Loading and filtering edges")
    edge_index = dataset.edge_index('paper', 'cites', 'paper')
    edge_index = torch.from_numpy(edge_index)
    
    # Filter edges to only include nodes in our subset
    node_indices_set = set(node_indices.numpy())
    mask = np.isin(edge_index[0].numpy(), node_indices_set) & np.isin(edge_index[1].numpy(), node_indices_set)
    edge_index = edge_index[:, mask]
    
    # Remap node ids to be consecutive
    print("Remapping node IDs")
    edge_index_remapped = torch.zeros_like(edge_index)
    for i in range(2):
        for j in tqdm(range(edge_index.size(1)), desc=f"Remapping edge index dim {i}"):
            edge_index_remapped[i, j] = node_mapping[edge_index[i, j].item()]
    
    torch.save(edge_index_remapped, edge_path)
    
    # Create PyG Data object
    data = Data(
        x=features,
        edge_index=edge_index_remapped,
        num_nodes=len(node_indices)
    )
    
    # Save the graph
    torch.save(data, graph_path)
    print(f"MAG240M subset saved to {graph_path}")
    
    return data, node_indices

def main():
    parser = argparse.ArgumentParser(description='Train GraphSAGE on MAG240M subset and save embeddings with matching indices')
    parser.add_argument('--root', type=str, default='./FSdatasets', help='Root directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./embeddings', help='Output directory for embeddings')
    parser.add_argument('--subset_size', type=int, default=20000000, help='Size of MAG240M subset')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--out_dim', type=int, default=256, help='Output embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GraphSAGE layers')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Evaluation steps')
    parser.add_argument('--save_steps', type=int, default=5000, help='Save model steps')
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[10, 10], 
                        help='Number of neighbors to sample at each layer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Set device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create or load MAG240M subset
    data, original_node_indices = create_mag240m_subset(args.root, args.subset_size, args.random_seed)
    
    # Keep data on CPU - DO NOT move to GPU
    print(f"Graph loaded with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
    print(f"Node features shape: {data.x.shape}")
    print(f"Original node indices range: min={original_node_indices.min().item()}, max={original_node_indices.max().item()}")
    
    # Create dataloader - now handles moving data to GPU in batches
    print("Creating data loaders")
    train_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # For embedding extraction - no shuffling to maintain original order
    embed_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    in_channels = data.x.size(1)
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, f'graphsage_best_model.pt')
    
    print(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        total_loss = 0
        processed_batches = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            # Move batch to device (not the whole graph)
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            node_embeddings = model(batch.x, batch.edge_index)
            
            # Compute loss - we'll use a simple link prediction objective
            src, dst = batch.edge_index
            
            # Filter edges to only use those where both nodes are in the batch
            edge_mask = (src < batch.num_nodes) & (dst < batch.num_nodes)
            src, dst = src[edge_mask], dst[edge_mask]
            
            if src.size(0) == 0:  # Skip if no edges in batch
                continue
                
            # Positive examples: real edges
            pos_score = torch.sum(node_embeddings[src] * node_embeddings[dst], dim=1)
            
            # Random negative edges
            neg_dst = torch.randint(0, batch.num_nodes, (src.size(0),), device=device)
            neg_score = torch.sum(node_embeddings[src] * node_embeddings[neg_dst], dim=1)
            
            # BPR loss
            loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-15))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            processed_batches += 1
            
            # Clear batch from GPU to save memory
            del batch, node_embeddings, src, dst, pos_score, neg_dst, neg_score
            torch.cuda.empty_cache()
            
            # Print progress
            if (i + 1) % args.eval_steps == 0:
                avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
                print(f'Epoch: {epoch+1}, Step: {i+1}, Loss: {avg_loss:.4f}')
                
                # Save best model
                if avg_loss < best_loss and processed_batches > 0:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), best_model_path)
                    print(f'New best model saved with loss: {best_loss:.4f}')
                
                total_loss = 0
                processed_batches = 0
            
            # Save checkpoint
            if (i + 1) % args.save_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f'graphsage_checkpoint_e{epoch+1}_s{i+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'step': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
    
    # Load best model
    print("Loading best model for embedding extraction")
    model.load_state_dict(torch.load(best_model_path))
    
    # Extract embeddings - ensuring indices match the subset node IDs
    print("Extracting embeddings for all nodes...")
    embeddings, id_mapping = model.get_embeddings(embed_loader, device, original_node_indices, args.out_dim)
    
    # Save embeddings and the mapping
    embedding_path = os.path.join(args.output_dir, f'mag240m_subset_{args.subset_size}_graphsage_embeddings.pt')
    mapping_path = os.path.join(args.output_dir, f'mag240m_subset_{args.subset_size}_id_mapping.pt')
    
    # Save the embeddings
    torch.save(embeddings, embedding_path)
    print(f"Embeddings saved to {embedding_path}")
    print(f"Embeddings shape: {embeddings.shape} - matches exactly with subset size")
    
    # Save the id mapping
    torch.save(id_mapping, mapping_path)
    print(f"ID mapping saved to {mapping_path}")
    
    # Also save a combined version for convenience
    combined_path = os.path.join(args.output_dir, f'mag240m_subset_{args.subset_size}_graphsage_embeddings_with_mapping.pt')
    torch.save({
        'embeddings': embeddings,
        'id_mapping': id_mapping,
        'original_node_indices': original_node_indices
    }, combined_path)
    print(f"Combined embeddings and mapping saved to {combined_path}")
    
    # Verify index alignment
    print("\nVerification of index alignment:")
    print(f"- Subset has {len(original_node_indices)} nodes")
    print(f"- Embeddings matrix has {embeddings.shape[0]} rows")
    print(f"- ID mapping has {len(id_mapping)} entries")
    print("- For any node at position i in the subset, its embedding is at position i in the embeddings matrix")
    print("- The ID mapping allows conversion from position i to original MAG240M node ID")
    
    print("\nTraining and embedding extraction completed!")

if __name__ == '__main__':
    main()