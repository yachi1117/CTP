import torch
import os
import sys
from tqdm import tqdm

# Add project root to path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import using absolute paths
from data.kg import get_kg_dataset

# Configuration
ROOT_PATH = "DATA_ROOT"  # Update with your actual path
OUTPUT_DIR = "./embeddings"
DATASET_NAME = "Wiki"

def extract_raw_embeddings():
    """Extract and save the raw embeddings from Wiki dataset"""
    print(f"Extracting raw embeddings for {DATASET_NAME} dataset...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    dataset = get_kg_dataset(
        root=ROOT_PATH, 
        name=DATASET_NAME, 
        n_hop=2,
        bert='sentence-transformers/all-mpnet-base-v2', 
        bert_device='cpu',
        node_graph=True
    )
    
    # Get the graph
    graph = dataset.pyg_graph
    
    # Extract existing node features (BERT embeddings)
    if hasattr(graph, 'x') and graph.x is not None:
        node_features = graph.x
        print(f"Extracted features with shape: {node_features.shape}")
        
        # Save features
        output_path = os.path.join(OUTPUT_DIR, f"{DATASET_NAME.lower()}_raw_embeddings.pt")
        torch.save(node_features, output_path)
        print(f"Saved raw embeddings to {output_path}")
        return node_features
    else:
        print("No node features found in the graph!")
        return None

if __name__ == "__main__":
    extract_raw_embeddings()