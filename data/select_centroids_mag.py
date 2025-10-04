import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import random
import gc
import math
import logging
import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def compute_kmeans_centroids_faiss(embeddings, args):
    """Compute K-means centroids using FAISS"""
    logger.info(f"Computing {args.num_centroids} centroids using FAISS K-means")
    
    # Sample data if needed
    if args.kmeans_sample_ratio < 1.0:
        sample_size = min(int(embeddings.shape[0] * args.kmeans_sample_ratio), args.max_kmeans_samples)
        logger.info(f"Sampling {sample_size} nodes for K-means")
        indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        if embeddings.shape[0] <= args.max_kmeans_samples:
            sample_embeddings = embeddings
        else:
            logger.info(f"Dataset too large. Sampling {args.max_kmeans_samples} nodes for K-means")
            indices = np.random.choice(embeddings.shape[0], args.max_kmeans_samples, replace=False)
            sample_embeddings = embeddings[indices]
    
    # Convert to numpy if it's a torch tensor
    if isinstance(sample_embeddings, torch.Tensor):
        sample_embeddings_np = sample_embeddings.numpy().astype(np.float32)
    else:
        sample_embeddings_np = sample_embeddings.astype(np.float32)
    
    # Create FAISS kmeans object
    d = sample_embeddings_np.shape[1]  # Dimensionality
    n_clusters = args.num_centroids
    
    # Use GPU if available
    use_gpu = hasattr(faiss, 'StandardGpuResources') and torch.cuda.is_available()
    
    if use_gpu:
        logger.info(f"Using GPU #{args.gpu_device} for FAISS clustering")
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = args.gpu_device  # Use specified GPU
        
        kmeans = faiss.Clustering(d, n_clusters)
        kmeans.niter = args.kmeans_max_iter
        kmeans.min_points_per_centroid = 1  # Min points per cluster
        kmeans.max_points_per_centroid = 1000000000  # Effectively no limit
        
        # Initialize centroids with k-means++
        index = faiss.GpuIndexFlatL2(res, d, config)
        kmeans.train(sample_embeddings_np, index)
        centroids = faiss.vector_float_to_array(kmeans.centroids)
    else:
        logger.info("Using CPU for FAISS clustering")
        kmeans = faiss.Kmeans(d, n_clusters, niter=args.kmeans_max_iter, 
                              verbose=True, seed=args.seed)
        kmeans.train(sample_embeddings_np)
        centroids = kmeans.centroids
    
    centroids = centroids.reshape(n_clusters, d)
    centroids_tensor = torch.from_numpy(centroids).float()
    
    return centroids_tensor, kmeans

def find_closest_nodes_to_centroids_faiss(embeddings, centroids, args):
    """Find the nodes closest to each centroid using FAISS"""
    logger.info(f"Finding closest nodes to {len(centroids)} centroids")
    
    # Convert centroids to numpy
    if isinstance(centroids, torch.Tensor):
        centroids_np = centroids.numpy().astype(np.float32)
    else:
        centroids_np = centroids.astype(np.float32)
    
    # Initialize array to store closest nodes and their distances
    # Use numpy arrays instead of torch tensors
    closest_nodes_np = np.full((len(centroids),), -1, dtype=np.int64)
    min_distances_np = np.full((len(centroids),), np.inf, dtype=np.float32)
    
    # Create a FAISS index for the centroids
    d = centroids_np.shape[1]  # Dimensionality
    
    # Use GPU if available
    use_gpu = hasattr(faiss, 'StandardGpuResources') and torch.cuda.is_available()
    
    if use_gpu:
        logger.info(f"Using GPU #{args.gpu_device} for closest node search")
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = args.gpu_device  # Use specified GPU
        centroid_index = faiss.GpuIndexFlatL2(res, d, config)
    else:
        centroid_index = faiss.IndexFlatL2(d)
    
    centroid_index.add(centroids_np)
    
    # Process embeddings in chunks
    num_nodes = embeddings.shape[0]
    chunk_size = args.chunk_size
    
    for chunk_idx in tqdm(range(0, num_nodes, chunk_size)):
        chunk_end = min(chunk_idx + chunk_size, num_nodes)
        chunk_data = embeddings[chunk_idx:chunk_end]
        
        if isinstance(chunk_data, torch.Tensor):
            chunk_data_np = chunk_data.numpy().astype(np.float32)
        else:
            chunk_data_np = chunk_data.astype(np.float32)
        
        # Skip invalid data
        if np.isnan(chunk_data_np).any() or np.isinf(chunk_data_np).any():
            continue
        
        # For each node, find closest centroid
        distances, indices = centroid_index.search(chunk_data_np, 1)
        
        # For each node, update the closest centroids
        for i in range(len(chunk_data_np)):
            centroid_idx = indices[i, 0]
            distance = distances[i, 0]
            
            if distance < min_distances_np[centroid_idx]:
                min_distances_np[centroid_idx] = distance
                closest_nodes_np[centroid_idx] = chunk_idx + i
    
    # Check if we found valid nodes for all centroids
    invalid_mask = closest_nodes_np == -1
    if np.any(invalid_mask):
        num_invalid = np.sum(invalid_mask)
        logger.warning(f"Failed to find valid nodes for {num_invalid} centroids. Using random nodes instead.")
        random_nodes = np.random.randint(0, num_nodes, num_invalid)
        closest_nodes_np[invalid_mask] = random_nodes
    
    # Convert back to torch tensors for saving
    closest_nodes = torch.from_numpy(closest_nodes_np).long()
    min_distances = torch.from_numpy(min_distances_np).float()
    
    return closest_nodes, min_distances

# Rest of the code remains unchanged
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Find diverse centroid nodes for MAG240M graph using k-means')
    
    # Main parameters
    parser.add_argument('--embeddings-path', type=str, required=True, 
                        help='Path to saved node embeddings (.pt file)')
    parser.add_argument('--output-path', type=str, default=None, 
                        help='Path to save centroid indices (default: centroids in same directory)')
    parser.add_argument('--num-centroids', type=int, default=30, help='Number of centroids (k)')
    
    # K-means parameters
    parser.add_argument('--kmeans-max-iter', type=int, default=100, help='Maximum iterations for KMeans')
    parser.add_argument('--kmeans-sample-ratio', type=float, default=1, 
                        help='Ratio of nodes to sample for KMeans (to save memory)')
    parser.add_argument('--max-kmeans-samples', type=int, default=20000000, 
                        help='Maximum number of samples to use for KMeans')
    
    # Processing parameters
    parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size for processing embeddings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--subset-size', type=int, default=20000000, 
                        help='Size of the MAG240M subset (should match the embeddings)')
    parser.add_argument('--gpu-device', type=int, default=3, help='GPU device ID to use (default: 3, the 4th GPU)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine output path
    if args.output_path is None:
        output_dir = os.path.dirname(args.embeddings_path)
        output_path = os.path.join(output_dir, f"mag240m_subset_{args.subset_size}_centroid_indices_k{args.num_centroids}.pt")
    else:
        output_path = args.output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Load embeddings
        logger.info(f"Loading embeddings from {args.embeddings_path}")
        
        try:
            data = torch.load(args.embeddings_path, map_location='cpu')
            
            # Check if this is a combined format with embeddings and mapping
            if isinstance(data, dict) and 'embeddings' in data:
                embeddings = data['embeddings']
                logger.info(f"Loaded combined format embeddings with shape: {embeddings.shape}")
                if 'id_mapping' in data:
                    logger.info("Found ID mapping in the embeddings file")
            # Check if it's just the embeddings tensor
            elif isinstance(data, torch.Tensor):
                embeddings = data
                logger.info(f"Loaded embeddings tensor with shape: {embeddings.shape}")
            else:
                logger.warning("Unrecognized data format, trying to use as embeddings directly")
                embeddings = data
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return 1
        
        # Verify embeddings size matches the expected subset size
        if embeddings.shape[0] != args.subset_size:
            logger.warning(f"Embeddings size ({embeddings.shape[0]}) doesn't match expected subset size ({args.subset_size})")
            logger.warning("This may cause issues with index alignment. Consider using the correct subset size.")
        
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        
        # Step 1: Compute centroids using K-means
        centroids, kmeans_model = compute_kmeans_centroids_faiss(embeddings, args)
        
        # Step 2: Find nodes closest to centroids
        closest_nodes, min_distances = find_closest_nodes_to_centroids_faiss(embeddings, centroids, args)
        
        # Step 3: Save indices (these are already the indices into the subset)
        logger.info(f"Saving centroid indices (values range from 0 to {embeddings.shape[0]-1})")
        torch.save(closest_nodes, output_path)
        logger.info(f"Saved {len(closest_nodes)} centroid indices to {output_path}")
        logger.info(f"Centroid indices range: min={closest_nodes.min().item()}, max={closest_nodes.max().item()}")
        
        # Print the actual centroid indices for inspection
        logger.info("Selected centroid indices (first 10 if more than 10):")
        logger.info(closest_nodes[:min(10, len(closest_nodes))].tolist())
        
        # Print usage instructions
        logger.info("\nTo use these centroids with the MAG240M subset, run:")
        logger.info(f"python experiments/run_single_experiment.py --dataset mag240m \\\n"
                   f"  --root <DATA_ROOT> \\\n"
                   f"  --original_features True \\\n"
                   f"  --centroids_path {output_path} \\\n"
                   f"  --n_way {args.num_centroids} \\\n"
                   f"  --n_shots 3 \\\n"
                   f"  --n_query 4 \\\n"
                   f"  --use_mag_subset True \\\n"
                   f"  --mag_subset_size {args.subset_size}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    main()