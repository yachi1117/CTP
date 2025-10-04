#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import argparse
import random
import sys
from collections import Counter
from tabulate import tabulate
from tqdm import tqdm
from contextlib import redirect_stdout
from datetime import datetime

def tee_print(*args, file_handle=None, **kwargs):
    """Print to both stdout and file"""
    print(*args, **kwargs)  # Print to stdout
    if file_handle:
        print(*args, file=file_handle, **kwargs)  # Print to file

def load_mag240m_subset(root_dir, subset_size=20000000, file_handle=None):
    """Load the MAG240M subset graph"""
    subset_path = os.path.join(root_dir, "mag240m", "mag240m_subset", f"mag240m_subset_{subset_size}.pt")
    tee_print(f"Loading MAG240M subset from {subset_path}", file_handle=file_handle)
    subset_graph = torch.load(subset_path)
    tee_print(f"Loaded graph with {subset_graph.num_nodes} nodes and {subset_graph.edge_index.size(1)} edges", file_handle=file_handle)
    return subset_graph

def load_centroids(centroids_path, file_handle=None):
    """Load the centroid indices"""
    tee_print(f"Loading centroids from {centroids_path}", file_handle=file_handle)
    centroids = torch.load(centroids_path)
    return centroids

def calculate_node_degrees(edge_index, num_nodes, node_indices=None, file_handle=None):
    """Calculate node degrees (number of connections) for specified nodes"""
    tee_print("Counting source nodes...", file_handle=file_handle)
    # Count how many times each node appears as a source node (outgoing edges)
    source_nodes = edge_index[0].numpy()
    source_degree_counter = Counter(source_nodes)
    
    tee_print("Counting target nodes...", file_handle=file_handle)
    # Count how many times each node appears as a target node (incoming edges)
    target_nodes = edge_index[1].numpy()
    target_degree_counter = Counter(target_nodes)
    
    tee_print("Calculating total degrees...", file_handle=file_handle)
    # Calculate total degree (in + out)
    total_degrees = {}
    all_nodes = set(source_degree_counter.keys()).union(set(target_degree_counter.keys()))
    for node in tqdm(all_nodes, desc="Processing node degrees"):
        total_degrees[node] = source_degree_counter.get(node, 0) + target_degree_counter.get(node, 0)
    
    # If node_indices are provided, return degrees for those specific nodes
    if node_indices is not None:
        tee_print(f"Retrieving degrees for {len(node_indices)} specific nodes...", file_handle=file_handle)
        degrees = []
        for idx in tqdm(node_indices, desc="Looking up node degrees"):
            degrees.append(total_degrees.get(idx.item(), 0))
        return degrees
    
    # Otherwise, return all degrees
    return total_degrees
    
def sample_random_nodes(num_nodes, sample_size, seed=42, file_handle=None):
    """Sample random nodes from the graph"""
    random.seed(seed)
    return torch.tensor(random.sample(range(num_nodes), sample_size))

def print_degree_distribution(centroid_degrees, random_degrees, file_handle=None):
    """Print a comparison of degree distributions"""
    # Calculate statistics
    centroid_mean = np.mean(centroid_degrees)
    centroid_median = np.median(centroid_degrees)
    centroid_min = np.min(centroid_degrees)
    centroid_max = np.max(centroid_degrees)
    
    random_mean = np.mean(random_degrees)
    random_median = np.median(random_degrees)
    random_min = np.min(random_degrees)
    random_max = np.max(random_degrees)
    
    # Print summary statistics
    tee_print("\n=== Degree Distribution Comparison ===", file_handle=file_handle)
    
    stats_table = [
        ["Statistic", "Centroid Nodes", "Random Nodes", "Ratio (C/R)"],
        ["Mean Degree", f"{centroid_mean:.2f}", f"{random_mean:.2f}", f"{centroid_mean/random_mean if random_mean > 0 else 'inf':.2f}x"],
        ["Median Degree", f"{centroid_median:.2f}", f"{random_median:.2f}", f"{centroid_median/random_median if random_median > 0 else 'inf':.2f}x"],
        ["Min Degree", f"{centroid_min:.0f}", f"{random_min:.0f}", "-"],
        ["Max Degree", f"{centroid_max:.0f}", f"{random_max:.0f}", "-"],
        ["Total Connections", f"{sum(centroid_degrees)}", f"{sum(random_degrees)}", 
         f"{sum(centroid_degrees)/sum(random_degrees) if sum(random_degrees) > 0 else 'inf':.2f}x"]
    ]
    
    stats_table_str = tabulate(stats_table, headers="firstrow", tablefmt="grid")
    tee_print(stats_table_str, file_handle=file_handle)
    
    # Print percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_table = [["Percentile", "Centroid Nodes", "Random Nodes", "Ratio (C/R)"]]
    
    for p in percentiles:
        c_perc = np.percentile(centroid_degrees, p)
        r_perc = np.percentile(random_degrees, p)
        ratio = c_perc / r_perc if r_perc > 0 else float('inf')
        percentile_table.append([f"{p}th", f"{c_perc:.0f}", f"{r_perc:.0f}", f"{ratio:.2f}x"])
    
    tee_print("\n=== Percentile Comparison ===", file_handle=file_handle)
    percentile_table_str = tabulate(percentile_table, headers="firstrow", tablefmt="grid")
    tee_print(percentile_table_str, file_handle=file_handle)
    
    # Print individual centroid degrees (limit to first 100 to avoid huge output)
    tee_print("\n=== Individual Centroid Node Degrees (first 100) ===", file_handle=file_handle)
    centroid_detail = [["Centroid #", "Degree"]]
    for i, degree in enumerate(centroid_degrees[:100]):
        centroid_detail.append([f"Centroid {i+1}", f"{degree}"]) 
    
    centroid_detail_str = tabulate(centroid_detail, headers="firstrow", tablefmt="grid")
    tee_print(centroid_detail_str, file_handle=file_handle)
    
    # Save full degree lists to a separate file
    tee_print("\nSaving full degree lists to separate CSV files...", file_handle=file_handle)
    np.savetxt('centroid_degrees.csv', centroid_degrees, delimiter=',', fmt='%d')
    np.savetxt('random_degrees.csv', random_degrees, delimiter=',', fmt='%d')
    tee_print("Degree lists saved.", file_handle=file_handle)

def main():
    parser = argparse.ArgumentParser(description='Analyze degree distribution of centroid nodes vs random nodes')
    
    parser.add_argument('--root', type=str, default='DATA_ROOT', 
                        help='Root directory for MAG240M dataset')
    parser.add_argument('--centroids-path', type=str, 
                        default='./embeddings/mag240m_subset_20000000_centroid_indices_k10000.pt',
                        help='Path to centroid indices file')
    parser.add_argument('--subset-size', type=int, default=20000000,
                        help='Size of the MAG240M subset')
    parser.add_argument('--random-samples', type=int, default=10000,
                        help='Number of random nodes to sample for comparison')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for node sampling')
    parser.add_argument('--output-file', type=str, 
                        default=f'degree_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                        help='Path to output file')
    
    args = parser.parse_args()
    
    # Open output file
    with open(args.output_file, 'w') as f:
        # Write header
        tee_print(f"=== Node Degree Analysis ===", file_handle=f)
        tee_print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file_handle=f)
        tee_print(f"Command line arguments:", file_handle=f)
        for arg, value in vars(args).items():
            tee_print(f"  {arg}: {value}", file_handle=f)
        tee_print("\n", file_handle=f)
        
        # Load the graph
        graph = load_mag240m_subset(args.root, args.subset_size, file_handle=f)
        
        # Load centroids
        centroids = load_centroids(args.centroids_path, file_handle=f)
        tee_print(f"Loaded {len(centroids)} centroid indices", file_handle=f)
        
        # Sample random nodes
        random_nodes = sample_random_nodes(graph.num_nodes, args.random_samples, args.seed, file_handle=f)
        tee_print(f"Sampled {len(random_nodes)} random nodes for comparison", file_handle=f)
        
        # Calculate node degrees
        tee_print("Calculating node degrees (this may take a while for a large graph)...", file_handle=f)
        centroid_degrees = calculate_node_degrees(graph.edge_index, graph.num_nodes, centroids, file_handle=f)
        random_degrees = calculate_node_degrees(graph.edge_index, graph.num_nodes, random_nodes, file_handle=f)
        
        # Print degree distributions
        print_degree_distribution(centroid_degrees, random_degrees, file_handle=f)
        
        tee_print("\nAnalysis complete!", file_handle=f)
        tee_print(f"Results saved to {args.output_file}", file_handle=f)

if __name__ == "__main__":
    main()