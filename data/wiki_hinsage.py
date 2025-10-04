import os
import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero  # Import to_hetero
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData # Import HeteroData
from tqdm import tqdm
import numpy as np
import gc
import copy # For deep copying data structure if needed

# Placeholder for specific Wiki dataset loading - REPLACE THIS
# from torch_geometric.datasets import WikipediaNetwork # Example
# Or your custom loading logic

# --- Placeholder Dataset Loading Function ---
# You MUST replace this with your actual dataset loading logic
# It should return a HeteroData object and optionally metadata
from torch_geometric.datasets import WikiCS
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import torch
import numpy as np
import random
import json

def load_wiki_dataset(root_dir):
    """
    Loads the Wiki dataset from PRODIGY and converts it to a HeteroData object.
    """
    print(f"Loading Wiki dataset from {root_dir}...")
    
    # Paths for Wiki dataset files
    wiki_dir = os.path.join(root_dir, "Wiki")
    entity2id_path = os.path.join(wiki_dir, "entity2id.json")
    relation2id_path = os.path.join(wiki_dir, "relation2id.json")
    path_graph_path = os.path.join(wiki_dir, "path_graph.json")
    text_features_path = os.path.join(wiki_dir, "text_features_web_scraped.pb")
    
    # Load entity and relation mappings
    with open(entity2id_path, 'r') as f:
        entity2id = json.load(f)
    
    with open(relation2id_path, 'r') as f:
        relation2id = json.load(f)
    
    # Create reverse mappings
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    
    # Load edges from path_graph.json
    with open(path_graph_path, 'r') as f:
        triples = json.load(f)
    
    # Create edge lists
    src_nodes = []
    dst_nodes = []
    edge_types = []
    
    for triple in triples:
        head, relation, tail = triple
        src_nodes.append(entity2id[head])
        dst_nodes.append(entity2id[tail])
        edge_types.append(relation2id[relation])
    
    # Create heterogeneous graph
    data = HeteroData()
    num_entities = len(entity2id)
    
    # Load text features with better error handling
    node_features = None
    try:
        import pickle
        print(f"Attempting to load text features from {text_features_path}")
        with open(text_features_path, 'rb') as f:
            text_features = pickle.load(f)
        
        print(f"Successfully loaded text features for {len(text_features)} entities")
        print(f"Sample text feature type: {type(next(iter(text_features.values())))}")
        
        # Create feature matrix with proper dimensions
        feature_dim = 128  # Default dimension
        
        # Try to determine feature dimension from the data
        for entity, vec in text_features.items():
            if isinstance(vec, (list, tuple, np.ndarray)) and len(vec) > 0:
                feature_dim = min(feature_dim, len(vec))
                break
            elif hasattr(vec, 'shape') and len(vec.shape) > 0:
                feature_dim = min(feature_dim, vec.shape[0])
                break
        
        print(f"Using feature dimension: {feature_dim}")
        node_features = torch.zeros((num_entities, feature_dim))
        
        # Keep track of mapped features
        mapped_count = 0
        
        # Fill features where available
        for entity_name, entity_id in entity2id.items():
            if entity_name in text_features:
                try:
                    feature_vec = text_features[entity_name]
                    # Convert to tensor if needed
                    if not isinstance(feature_vec, torch.Tensor):
                        if isinstance(feature_vec, (list, tuple, np.ndarray)):
                            feature_vec = torch.tensor(feature_vec[:feature_dim], dtype=torch.float)
                        else:
                            # Skip if feature vector is not a recognized type
                            continue
                    
                    # Ensure correct shape and size
                    if len(feature_vec) >= feature_dim:
                        node_features[entity_id] = feature_vec[:feature_dim]
                        mapped_count += 1
                except Exception as e:
                    print(f"Error processing feature for entity {entity_name}: {e}")
                    continue
        
        print(f"Successfully mapped features for {mapped_count}/{num_entities} entities")
        
    except Exception as e:
        print(f"Failed to load or process text features: {e}")
        node_features = None
    
    # Fallback to random features if needed
    if node_features is None:
        print("Using random features as fallback")
        node_features = torch.randn((num_entities, 128))
    
    # Add nodes and features to the graph
    data['article'].x = node_features
    data['article'].num_nodes = num_entities
    
    # Add article-to-article edges for each relation type
    relation_edge_indices = {}
    
    for src, dst, rel in zip(src_nodes, dst_nodes, edge_types):
        rel_name = id2relation[rel]
        if ('article', rel_name, 'article') not in relation_edge_indices:
            relation_edge_indices[('article', rel_name, 'article')] = [[], []]
        
        relation_edge_indices[('article', rel_name, 'article')][0].append(src)
        relation_edge_indices[('article', rel_name, 'article')][1].append(dst)
    
    # Add edges to the graph
    for edge_type, edge_list in relation_edge_indices.items():
        data[edge_type].edge_index = torch.tensor(edge_list, dtype=torch.long)
    
    # Add generic 'links' relation 
    all_src = torch.tensor(src_nodes, dtype=torch.long)
    all_dst = torch.tensor(dst_nodes, dtype=torch.long)
    data['article', 'links', 'article'].edge_index = torch.stack([all_src, all_dst], dim=0)
    
    print(f"Wiki graph loaded with {num_entities} entities and {len(src_nodes)} edges")
    print(f"Node feature dimensions: {data['article'].x.shape}")
    print(f"Edge types: {data.edge_types}")
    
    return data

class GraphSAGE(torch.nn.Module):
    # Base Homogeneous GraphSAGE model (will be converted by to_hetero)
    def __init__(self, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        # Note: We don't need in_channels here if using lazy initialization with to_hetero
        #       SAGEConv with -1 will infer input sizes.

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        # Input layer (using -1 for lazy init)
        # SAGEConv needs (-1, -1) for heterogeneous input shapes
        self.convs.append(SAGEConv((-1, -1), hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv((-1, -1), hidden_channels))

        # Output layer
        self.convs.append(SAGEConv((-1, -1), out_channels))

    def forward(self, x, edge_index):
        # This forward is for the *homogeneous* case.
        # to_hetero will adapt this logic for heterogeneous inputs.
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training) # Keep dropout consistent
        return x

# Modified get_embeddings for HeteroData and target node type
def get_embeddings(model, data_loader, device, primary_node_type, original_primary_node_indices, embedding_dim):
    """
    Extract embeddings for a specific node type while preserving the mapping.

    Args:
        model: The trained *heterogeneous* GNN model.
        data_loader: NeighborLoader for the heterogeneous graph subset.
        device: Device to run inference on.
        primary_node_type: The name of the node type to extract embeddings for.
        original_primary_node_indices: Tensor of original node IDs (before subsetting)
                                         for the primary_node_type.
        embedding_dim: Dimension of the embeddings.

    Returns:
        embeddings: Tensor of shape [num_primary_subset_nodes, embedding_dim]
        id_mapping: Mapping dictionary from position in embeddings to original node ID.
    """
    model.eval()

    # Create mapping from remapped indices (0 to subset_size-1 for primary type)
    # to original node indices of that type
    num_primary_nodes_in_subset = len(original_primary_node_indices)
    remapped_to_original = {
        i: original_id.item() for i, original_id in enumerate(original_primary_node_indices)
    }

    # Initialize embeddings tensor - using the SUBSET size for the primary node type
    all_embeddings = torch.zeros(num_primary_nodes_in_subset, embedding_dim, device='cpu')

    # Track processed nodes (using subset indices)
    processed_nodes = set()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Extracting embeddings for '{primary_node_type}'"):
            batch = batch.to(device)

            # Run heterogeneous model
            batch_embeddings_dict = model(batch.x_dict, batch.edge_index_dict)

            # Get embeddings for the primary node type
            batch_primary_embeddings = batch_embeddings_dict[primary_node_type]

            # Get the original (subset-level) indices for the central nodes of the primary type in the batch
            # These are the indices within the range [0, num_primary_nodes_in_subset - 1]
            # batch.batch_size is not directly applicable in HeteroData like this for a specific type.
            # We need the node IDs of the *input nodes* for this batch.
            # NeighborLoader stores the original indices of the seed nodes in batch[node_type].n_id[:batch[node_type].batch_size]
            # Note: batch[node_type].batch_size stores the number of seed nodes of that type for this batch.

            # Ensure the primary node type was actually sampled as input/seed node
                  
            # Get the original (subset-level) indices for the central nodes of the primary type in the batch
            # These are the indices within the range [0, num_primary_nodes_in_subset - 1]
            # NeighborLoader puts the seed nodes first for each type.
            # batch[node_type].batch_size stores the number of seed nodes of that type for this batch.

            # Check if the primary node type is present in the batch at all and has the necessary attributes
            if primary_node_type in batch.node_types and hasattr(batch[primary_node_type], 'batch_size') and hasattr(batch[primary_node_type], 'n_id'):
                 num_seed_nodes = batch[primary_node_type].batch_size
                 if num_seed_nodes > 0:
                    # The first `num_seed_nodes` in the batch correspond to the input seeds.
                    # Their original indices (within the subset graph) are the first `num_seed_nodes` of `n_id`.
                    # Ensure n_id has enough elements before slicing
                    if batch[primary_node_type].n_id.shape[0] >= num_seed_nodes:
                        batch_indices = batch[primary_node_type].n_id[:num_seed_nodes].cpu().numpy()

                        # Store embeddings at the EXACT same positions as their index in the subset
                        # The model output (batch_primary_embeddings) corresponds to all nodes of that type in the batch.
                        # The first 'num_seed_nodes' embeddings correspond to the seed nodes.
                        for i, subset_idx in enumerate(batch_indices):
                             # Double-check index is valid for the pre-allocated all_embeddings tensor
                             if subset_idx < all_embeddings.shape[0]:
                                 all_embeddings[subset_idx] = batch_primary_embeddings[i].cpu()
                                 processed_nodes.add(subset_idx)
                             else:
                                 print(f"Warning: Subset index {subset_idx} out of bounds for all_embeddings (size {all_embeddings.shape[0]}). This might indicate an issue.")
                    else:
                        print(f"Warning: Mismatch between batch_size ({num_seed_nodes}) and n_id size ({batch[primary_node_type].n_id.shape[0]}) for type '{primary_node_type}'. Skipping batch.")

            elif primary_node_type in batch.node_types:
                 # Node type exists, but missing batch_size or n_id? This would be unusual.
                 print(f"Warning: Node type '{primary_node_type}' found in batch, but missing 'batch_size' or 'n_id' attribute. Skipping batch for embedding extraction.")
            # else: The primary node type wasn't even included in this batch's computation graph (e.g., if it wasn't reachable).

    

            del batch, batch_embeddings_dict, batch_primary_embeddings
            torch.cuda.empty_cache()
            gc.collect() # Force garbage collection

    # Verify that we have embeddings for all nodes in the subset
    expected_nodes = set(range(num_primary_nodes_in_subset))
    missing_nodes = expected_nodes - processed_nodes
    if missing_nodes:
        print(f"Warning: {len(missing_nodes)} nodes of type '{primary_node_type}' are missing embeddings")
        # This might happen if some nodes were not sampled as seeds by the loader.
        # Consider using input_nodes=torch.arange(num_primary_nodes_in_subset) in embed_loader
        # if you absolutely need all embeddings, but this might be slow.
    else:
        print(f"Successfully extracted embeddings for all {len(processed_nodes)} '{primary_node_type}' nodes")
        print(f"Embedding indices match exactly with the subset node indices (0 to {num_primary_nodes_in_subset-1}) for '{primary_node_type}'")

    return all_embeddings, remapped_to_original

# Modified subset creation for HeteroData
def create_wiki_subset(root_dir, primary_node_type, edge_types_to_keep, subset_size=100000, random_seed=42):
    """
    Create a subset of the Wiki dataset focusing on a primary node type.
    Includes only the sampled primary nodes and edges between them from the specified edge types.
    """
    subset_dir = os.path.join(root_dir, 'wiki_subset')
    os.makedirs(subset_dir, exist_ok=True)

    # Define paths based on primary type and size
    subset_graph_path = os.path.join(subset_dir, f'wiki_{primary_node_type}_subset_{subset_size}_graph.pt')
    subset_nodes_path = os.path.join(subset_dir, f'wiki_{primary_node_type}_subset_{subset_size}_nodes.pt')

    if os.path.exists(subset_graph_path) and os.path.exists(subset_nodes_path):
        print(f"Loading existing Wiki subset from {subset_graph_path}")
        data = torch.load(subset_graph_path)
        original_primary_node_indices = torch.load(subset_nodes_path)
        print(f"Loaded subset graph: {data}")
        return data, original_primary_node_indices

    print(f"Creating Wiki subset with {subset_size} nodes of type '{primary_node_type}'")

    # --- Load the full dataset ---
    full_data = load_wiki_dataset(root_dir) # Replace with your loading logic
    # ---

    if primary_node_type not in full_data.node_types:
        raise ValueError(f"Primary node type '{primary_node_type}' not found in dataset.")

    num_primary_nodes_full = full_data[primary_node_type].num_nodes
    if subset_size > num_primary_nodes_full:
        print(f"Warning: subset_size ({subset_size}) > num_primary_nodes ({num_primary_nodes_full}). Using all nodes.")
        subset_size = num_primary_nodes_full
        original_primary_node_indices = torch.arange(num_primary_nodes_full)
    else:
        np.random.seed(random_seed)
        original_primary_node_indices = np.random.choice(num_primary_nodes_full, subset_size, replace=False)
        original_primary_node_indices = np.sort(original_primary_node_indices)
        original_primary_node_indices = torch.tensor(original_primary_node_indices)

    torch.save(original_primary_node_indices, subset_nodes_path)

    # --- Create the subset HeteroData object ---
    subset_data = HeteroData()

    # 1. Add nodes and features for the primary type
    print(f"Adding features for {len(original_primary_node_indices)} '{primary_node_type}' nodes")
    if hasattr(full_data[primary_node_type], 'x') and full_data[primary_node_type].x is not None:
         subset_data[primary_node_type].x = full_data[primary_node_type].x[original_primary_node_indices]
    else:
         print(f"Warning: No features found for node type '{primary_node_type}'. Model might need adaptation.")
         # Add dummy features or handle featureless nodes in the model/training
         # subset_data[primary_node_type].x = torch.ones((len(original_primary_node_indices), 1)) # Example
    subset_data[primary_node_type].num_nodes = len(original_primary_node_indices)
    # Store original IDs if needed later (e.g., for mapping back results)
    subset_data[primary_node_type].original_ids = original_primary_node_indices

    # Create mapping from original full-graph ID to new subset ID (0 to N-1) for primary nodes
    primary_node_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(original_primary_node_indices)}
    primary_node_set = set(original_primary_node_indices.numpy())

    # 2. Filter and remap edges *between* the sampled primary nodes
    print("Filtering and remapping edges...")
    for edge_type in edge_types_to_keep:
        src_type, rel_type, dst_type = edge_type
        print(f"Processing edge type: {edge_type}")

        if edge_type not in full_data.edge_types:
            print(f"  Warning: Edge type {edge_type} not found in full dataset. Skipping.")
            continue

        # Ensure this edge type connects the primary node type to itself
        if src_type != primary_node_type or dst_type != primary_node_type:
            print(f"  Warning: Skipping edge type {edge_type} because it doesn't connect '{primary_node_type}' to itself. Modify 'edge_types_to_keep' or subset logic if needed.")
            continue

        full_edge_index = full_data[edge_type].edge_index
        
        # Mask to keep edges where *both* endpoints are in the sampled primary node set
        mask = np.isin(full_edge_index[0].numpy(), primary_node_set) & \
               np.isin(full_edge_index[1].numpy(), primary_node_set)

        subset_edge_index = full_edge_index[:, mask]
        print(f"  Found {subset_edge_index.size(1)} edges after filtering.")

        if subset_edge_index.size(1) > 0:
            # Remap edge indices to the new subset indices (0 to N-1)
            remapped_edge_index = torch.zeros_like(subset_edge_index)
            for i in range(2): # Source and destination
                original_ids_tensor = subset_edge_index[i]
                # Vectorized mapping is much faster than looping
                remapped_ids = torch.tensor([primary_node_mapping[idx.item()] for idx in original_ids_tensor], dtype=torch.long)
                remapped_edge_index[i] = remapped_ids

            subset_data[edge_type].edge_index = remapped_edge_index

            # Copy edge attributes if they exist
            if hasattr(full_data[edge_type], 'edge_attr') and full_data[edge_type].edge_attr is not None:
                 subset_data[edge_type].edge_attr = full_data[edge_type].edge_attr[mask]
        else:
             # Ensure the edge type exists in the graph even if empty
             subset_data[edge_type].edge_index = torch.empty((2,0), dtype=torch.long)


    # Clean up large objects
    del full_data
    gc.collect()

    # Optional: Add reverse edges if the original graph was directed and model expects undirected
    # print("Applying T.ToUndirected to the subset...")
    # subset_data = T.ToUndirected()(subset_data)

    print(f"Final subset graph structure:\n{subset_data}")
    torch.save(subset_data, subset_graph_path)
    print(f"Wiki subset graph saved to {subset_graph_path}")

    return subset_data, original_primary_node_indices

def main():
    parser = argparse.ArgumentParser(description='Train HinSAGE on Wiki subset and save embeddings')
    parser.add_argument('--root', type=str, default='./datasets', help='Root directory for datasets') # Changed default
    parser.add_argument('--output_dir', type=str, default='./embeddings_wiki', help='Output directory for embeddings') # Changed default
    parser.add_argument('--primary_node_type', type=str, default='article', help='The main node type for subsetting and embedding') # New arg
    # Adjust edge type based on your dataset, e.g., ('article', 'links', 'article')
    parser.add_argument('--target_edge_type', type=lambda s: tuple(s.split(',')), default='article,links,article',
                        help='The target edge type for training loss (comma-separated string: src,rel,dst)') # New arg
    parser.add_argument('--subset_size', type=int, default=100000, help='Size of Wiki subset based on primary node type') # Smaller default
    parser.add_argument('--device', type=int, default=0, help='GPU device ID (-1 for CPU)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size') # Smaller default
    parser.add_argument('--out_dim', type=int, default=128, help='Output embedding dimension') # Smaller default
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GraphSAGE layers')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for NeighborLoader') # Smaller default
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs') # Maybe more epochs needed
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate') # Adjusted default
    parser.add_argument('--eval_steps', type=int, default=200, help='Evaluation steps') # Adjusted default
    parser.add_argument('--save_steps', type=int, default=1000, help='Save model steps') # Adjusted default
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[10, 10],
                        help='Number of neighbors to sample at each layer')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers') # Adjusted default
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Validate target_edge_type format
    if len(args.target_edge_type) != 3:
        parser.error("--target_edge_type must be a comma-separated string of three parts: src,rel,dst")
    target_edge_type = args.target_edge_type # Keep as tuple

    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Set device
    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Create or load Wiki subset ---
    # Define which edge types to include in the subset graph structure.
    # This simplified version only keeps edges between primary nodes.
    # Adapt this list based on your needs and dataset structure.
    edge_types_to_keep_in_subset = [target_edge_type]
    # Add other ('primary', 'rel', 'primary') types if they exist and are relevant
    # edge_types_to_keep_in_subset.append(('article', 'other_link', 'article'))

    data, original_primary_node_indices = create_wiki_subset(
    args.root,
    args.primary_node_type,
    edge_types_to_keep_in_subset,
    args.subset_size,
    args.random_seed
    )

    print(f"Subset HeteroData loaded:\n{data}")
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    print(f"Number of primary nodes ('{args.primary_node_type}') in subset: {data[args.primary_node_type].num_nodes}")
    print(f"Original primary node indices range: min={original_primary_node_indices.min().item()}, max={original_primary_node_indices.max().item()}")

    # --- Create DataLoaders ---
    print("Creating data loaders...")
    # Note: NeighborLoader works with HeteroData directly.
    # We sample neighborhoods starting from the primary node type for training.
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: args.num_neighbors for key in data.edge_types}, # Sample neighbors for all edge types
        batch_size=args.batch_size,
        input_nodes=args.primary_node_type, # Sample starting from this node type
        shuffle=True,
        num_workers=args.num_workers
    )

    # Loader for embedding extraction - sample all primary nodes without shuffling
    # Use input_nodes=torch.arange(...) if you need *all* embeddings guaranteed,
    # but might be slower and need more memory if graph is large.
    # Sampling starting from primary_node_type usually covers most nodes eventually.
    embed_loader = NeighborLoader(
        data,
        num_neighbors={key: args.num_neighbors for key in data.edge_types},
        batch_size=args.batch_size, # Can use larger batch size for inference
        input_nodes=args.primary_node_type,
        shuffle=False,
        num_workers=args.num_workers
    )

    # --- Initialize Model ---
    # 1. Create the base homogeneous model
    base_model = GraphSAGE(
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        num_layers=args.num_layers
    )

    # 2. Convert it to a heterogeneous model using to_hetero
    # Pass the metadata (node types, edge types) of the subset graph
    model = to_hetero(base_model, data.metadata(), aggr='sum').to(device)

    print(f"Heterogeneous model created using to_hetero.")
    # Initialize lazy parameters by doing a dummy forward pass
    print("Initializing lazy parameters...")
    with torch.no_grad():
        # Get a sample batch
        init_batch = next(iter(train_loader)).to(device)
        model(init_batch.x_dict, init_batch.edge_index_dict)
        del init_batch # Free memory
        torch.cuda.empty_cache()
    print(f"Model initialized with ~{sum(p.numel() for p in model.parameters())} parameters.")


    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    model.train()
    best_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, f'hinsage_{args.primary_node_type}_best_model.pt')

    # Define source and destination types for the training loss edge type
    loss_src_type, _, loss_dst_type = target_edge_type
    if loss_src_type not in data.node_types or loss_dst_type not in data.node_types:
         raise ValueError(f"Source ('{loss_src_type}') or Destination ('{loss_dst_type}') node type "
                          f"for target_edge_type {target_edge_type} not found in graph subset.")
    if target_edge_type not in data.edge_types:
         print(f"Warning: Target edge type {target_edge_type} for loss calculation "
               f"is not present in the created subset graph. Training might not work.")


    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        total_loss = 0
        processed_batches = 0
        model.train() # Ensure model is in train mode

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass through the heterogeneous model
            out_dict = model(batch.x_dict, batch.edge_index_dict)

            # --- Unsupervised Link Prediction Loss (BPR) on target_edge_type ---
            loss = torch.tensor(0.0, device=device) # Initialize loss for the batch

            # Check if the target edge type exists in this batch
            if target_edge_type in batch.edge_types and batch[target_edge_type].edge_index.numel() > 0:
                # Get edges for the target type from the batch
                edge_index = batch[target_edge_type].edge_index
                src, dst = edge_index

                # Get embeddings for the source and destination node types involved
                # Note: these embeddings correspond to nodes *present in the current batch*
                src_emb = out_dict[loss_src_type]
                dst_emb = out_dict[loss_dst_type]

                # Positive examples: scores for existing edges
                # Indices (src, dst) are local to the nodes *of that type* within the batch
                pos_score = torch.sum(src_emb[src] * dst_emb[dst], dim=1)

                # Negative examples: sample random destination nodes *of the destination type*
                # Sample within the number of nodes of the destination type *present in this batch*
                num_dst_nodes_in_batch = batch[loss_dst_type].num_nodes
                neg_dst = torch.randint(0, num_dst_nodes_in_batch, (src.size(0),), device=device)
                neg_score = torch.sum(src_emb[src] * dst_emb[neg_dst], dim=1)

                # BPR Loss for this edge type
                type_loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-15))
                loss += type_loss

            if torch.is_grad_enabled() and loss.requires_grad: # Check if loss requires grad before backward()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                processed_batches += 1
            elif loss.item() == 0.0:
                 # print("Skipping batch - no target edges found or loss is zero.")
                 pass # Just continue if no target edges were in the batch
            else:
                 print(f"Warning: Loss does not require grad. Loss value: {loss.item()}")


            # Clear batch from GPU
            del batch, out_dict
            if 'edge_index' in locals(): del edge_index, src, dst, src_emb, dst_emb, pos_score, neg_dst, neg_score, type_loss
            del loss
            torch.cuda.empty_cache()
            # gc.collect() # Might slow down training if used every batch

            # Print progress and save model
            current_step = i + 1
            if current_step % args.eval_steps == 0 and processed_batches > 0:
                avg_loss = total_loss / processed_batches
                print(f'Epoch: {epoch+1}, Step: {current_step}, Avg Loss: {avg_loss:.4f}')

                # Save best model based on training loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), best_model_path)
                    print(f'New best model saved with loss: {best_loss:.4f}')

                total_loss = 0
                processed_batches = 0 # Reset for next evaluation interval

            if current_step % args.save_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f'hinsage_{args.primary_node_type}_checkpoint_e{epoch+1}_s{current_step}.pt')
                torch.save({
                    'epoch': epoch,
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss if processed_batches > 0 else float('nan') # Use last avg loss
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

    # --- Load Best Model and Extract Embeddings ---
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for embedding extraction")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: No best model found. Using the model from the end of training.")

    print(f"Extracting embeddings for node type '{args.primary_node_type}'...")
    embeddings, id_mapping = get_embeddings(
        model,
        embed_loader,
        device,
        args.primary_node_type,
        original_primary_node_indices,
        args.out_dim
    )

    # --- Save Embeddings and Mapping ---
    embedding_path = os.path.join(args.output_dir, f'wiki_{args.primary_node_type}_subset_{args.subset_size}_hinsage_embeddings.pt')
    mapping_path = os.path.join(args.output_dir, f'wiki_{args.primary_node_type}_subset_{args.subset_size}_id_mapping.pt')
    combined_path = os.path.join(args.output_dir, f'wiki_{args.primary_node_type}_subset_{args.subset_size}_hinsage_embeddings_with_mapping.pt')

    torch.save(embeddings, embedding_path)
    print(f"Embeddings saved to {embedding_path}")
    print(f"Embeddings shape: {embeddings.shape}")

    torch.save(id_mapping, mapping_path)
    print(f"ID mapping saved to {mapping_path}")

    torch.save({
        'embeddings': embeddings,
        'id_mapping': id_mapping,
        'original_node_indices': original_primary_node_indices, # Original IDs corresponding to subset indices
        'primary_node_type': args.primary_node_type
    }, combined_path)
    print(f"Combined embeddings and mapping saved to {combined_path}")

    # --- Verification ---
    print("\nVerification of index alignment:")
    print(f"- Subset created for primary node type: '{args.primary_node_type}'")
    print(f"- Number of '{args.primary_node_type}' nodes in subset: {len(original_primary_node_indices)}")
    print(f"- Embeddings matrix has {embeddings.shape[0]} rows")
    print(f"- ID mapping has {len(id_mapping)} entries")
    assert embeddings.shape[0] == len(original_primary_node_indices), "Mismatch in embedding rows and number of subset nodes"
    assert len(id_mapping) == len(original_primary_node_indices), "Mismatch in mapping size and number of subset nodes"
    print("- For any primary node at position i in the subset graph, its embedding is at row i in the embeddings matrix.")
    print("- The ID mapping allows conversion from row index i to the original Wiki node ID for that primary node.")

    # Example verification:
    example_idx = 0
    subset_node_idx = example_idx # Position in the subset
    original_node_id = id_mapping[subset_node_idx]
    embedding_vector = embeddings[subset_node_idx]
    print(f"- Example: Embedding at row {subset_node_idx} corresponds to original node ID {original_node_id} of type '{args.primary_node_type}'. Embedding shape: {embedding_vector.shape}")

    print("\nTraining and embedding extraction completed!")

if __name__ == '__main__':
    main()