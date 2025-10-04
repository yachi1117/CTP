import os
import argparse
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
from tqdm import tqdm

def find_optimal_k(embeddings, k_range, method='elbow'):
    """
    Find the optimal number of clusters using either the elbow method or silhouette score.
    
    Args:
        embeddings: Node embeddings
        k_range: Range of k values to try
        method: 'elbow' or 'silhouette'
        
    Returns:
        optimal_k: The optimal number of clusters
    """
    print(f"Finding optimal k using {method} method...")
    
    # For elbow method
    wcss = []  # Within-Cluster Sum of Squares
    
    # For silhouette method
    silhouette_scores = []
    
    # Try different values of k
    for k in tqdm(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        
        # For elbow method
        wcss.append(kmeans.inertia_)
        
        # For silhouette method (if k > 1)
        if k > 1:
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(embeddings, labels))
    
    # Plot WCSS vs k for elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.savefig('elbow_method_plot.png')
    plt.close()
    
    # Find the elbow point using the KneeLocator
    knee = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
    optimal_k_elbow = knee.elbow
    
    # Plot Silhouette score vs k (if k_range includes values > 1)
    if len(silhouette_scores) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(k_range[1:], silhouette_scores, 'ro-')
        plt.title('Silhouette Method for Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.savefig('silhouette_method_plot.png')
        plt.close()
        
        # Find the k with highest silhouette score
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores) + 1]  # +1 because silhouette starts from k=2
    else:
        optimal_k_silhouette = None
    
    # Return based on method
    if method == 'elbow':
        print(f"Optimal k using elbow method: {optimal_k_elbow}")
        return optimal_k_elbow
    else:
        print(f"Optimal k using silhouette method: {optimal_k_silhouette}")
        return optimal_k_silhouette

def get_cluster_centers(embeddings, n_clusters, return_type='indices'):
    """
    Perform k-means clustering and return either the centroid indices or the centroid embeddings.
    
    Args:
        embeddings: Node embeddings
        n_clusters: Number of clusters
        return_type: 'indices' to get indices of nodes closest to centroids, 'centroids' to get centroid embeddings
        
    Returns:
        Either indices of nodes closest to centroids or centroid embeddings
    """
    print(f"Performing k-means clustering with k={n_clusters}...")
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    if return_type == 'centroids':
        # Return the centroid embeddings
        return kmeans.cluster_centers_, cluster_labels
    else:
        # Find the node closest to each centroid
        centroids = kmeans.cluster_centers_
        centroid_indices = []
        
        for i in range(n_clusters):
            # Get indices of nodes in this cluster
            cluster_points = embeddings[cluster_labels == i]
            cluster_indices = np.where(cluster_labels == i)[0]
            
            # Calculate distance from each point to centroid
            distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
            
            # Get index of point closest to centroid
            closest_point_idx = cluster_indices[np.argmin(distances)]
            centroid_indices.append(closest_point_idx)
        
        return np.array(centroid_indices), cluster_labels

def select_query_nodes(embeddings, cluster_labels, n_per_cluster):
    """
    Select query nodes from each cluster for evaluation.
    
    Args:
        embeddings: Node embeddings
        cluster_labels: Cluster assignment for each node
        n_per_cluster: Number of nodes to select per cluster
        
    Returns:
        query_indices: Indices of selected query nodes
    """
    n_clusters = len(np.unique(cluster_labels))
    query_indices = []
    
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        
        # If the cluster has fewer nodes than requested, take all
        if len(cluster_indices) <= n_per_cluster:
            query_indices.extend(cluster_indices)
        else:
            # Randomly select n_per_cluster nodes from this cluster
            selected = np.random.choice(cluster_indices, n_per_cluster, replace=False)
            query_indices.extend(selected)
    
    return np.array(query_indices)

def main():
    parser = argparse.ArgumentParser(description='Perform k-means clustering on node embeddings')
    parser.add_argument('--embeddings_path', type=str, required=True, 
                        help='Path to the node embeddings file')
    parser.add_argument('--output_dir', type=str, default='./centroids', 
                        help='Output directory for centroids')
    # Note: The following parameters are kept for reference but not used 
    # since we're using a fixed k=30
    parser.add_argument('--min_k', type=int, default=5, 
                        help='Minimum number of clusters to try (not used with fixed k)')
    parser.add_argument('--max_k', type=int, default=50, 
                        help='Maximum number of clusters to try (not used with fixed k)')
    parser.add_argument('--step_k', type=int, default=5, 
                        help='Step size for k values (not used with fixed k)')
    parser.add_argument('--method', type=str, default='elbow', choices=['elbow', 'silhouette'],
                        help='Method to find optimal k (not used with fixed k)')
    parser.add_argument('--specified_k', type=int, default=30,
                        help='Fixed number of clusters to use')
    parser.add_argument('--query_per_cluster', type=int, default=10,
                        help='Number of query nodes to select per cluster')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings_path}...")
    embeddings = torch.load(args.embeddings_path)
    
    # Convert to numpy for sklearn
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Determine the number of clusters (k)
    # Using a fixed k of 30 as requested
    optimal_k = 30
    print(f"Using fixed k: {optimal_k}")
    
    # Uncomment below to find optimal k using the elbow method
    # if args.specified_k is not None:
    #     optimal_k = args.specified_k
    #     print(f"Using specified k: {optimal_k}")
    # else:
    #     k_range = list(range(args.min_k, args.max_k + 1, args.step_k))
    #     optimal_k = find_optimal_k(embeddings, k_range, method=args.method)
    
    # Get cluster centroids and labels
    centroid_indices, cluster_labels = get_cluster_centers(embeddings, optimal_k)
    
    # Save centroid indices
    centroids_path = os.path.join(args.output_dir, f'centroid_indices_k{optimal_k}.pt')
    torch.save(torch.tensor(centroid_indices, dtype=torch.long), centroids_path)
    print(f"Saved {len(centroid_indices)} centroid indices to {centroids_path}")
    
    # Save the cluster labels for all nodes
    labels_path = os.path.join(args.output_dir, f'cluster_labels_k{optimal_k}.pt')
    torch.save(torch.tensor(cluster_labels, dtype=torch.long), labels_path)
    print(f"Saved cluster labels for all nodes to {labels_path}")
    
    # Select and save query nodes
    query_indices = select_query_nodes(embeddings, cluster_labels, args.query_per_cluster)
    query_path = os.path.join(args.output_dir, f'query_indices_k{optimal_k}_q{args.query_per_cluster}.pt')
    torch.save(torch.tensor(query_indices, dtype=torch.long), query_path)
    print(f"Saved {len(query_indices)} query indices to {query_path}")
    
    # Print cluster statistics
    print("\nCluster statistics:")
    for i in range(optimal_k):
        cluster_size = (cluster_labels == i).sum()
        print(f"Cluster {i}: {cluster_size} nodes ({cluster_size/len(embeddings)*100:.2f}%)")
    
    # Create a visualization of cluster distribution
    plt.figure(figsize=(12, 6))
    cluster_sizes = [(cluster_labels == i).sum() for i in range(optimal_k)]
    plt.bar(range(optimal_k), cluster_sizes)
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Nodes')
    plt.savefig(os.path.join(args.output_dir, 'cluster_distribution.png'))
    plt.close()
    
    print("\nDone!")

if __name__ == '__main__':
    main()