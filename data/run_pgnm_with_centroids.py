#!/usr/bin/env python3
"""
Runner script for PG-NM with centroid-based node selection
"""
import os
import sys
import torch
import argparse
import numpy as np
import copy
import random

# Make sure PRODIGY code is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.params import get_params
from experiments.trainer import TrainerFS
from data.data_loader_wrapper import get_dataset_wrap


class CentroidNodeSampler:
    """
    A node sampler that uses precomputed centroid nodes for neighbor matching tasks
    """
    def __init__(self, centroid_nodes, neighbor_sampler):
        self.centroid_nodes = centroid_nodes
        self.neighbor_sampler = neighbor_sampler
        
    def sample(self, num_label, num_member, num_shot, num_query, rng):
        """
        Sample tasks using the centroid nodes instead of random nodes
        """
        task = {}
        # Use up to num_label centroids
        selected_centroids = self.centroid_nodes[:num_label]
        
        for center in selected_centroids:
            # Create a tensor with the center node repeated
            node_idx = torch.ones(num_member * 10, dtype=torch.long) * center
            
            # Sample neighbors through random walks
            node_idx = self.neighbor_sampler.random_walk(node_idx, "inout")
            node_idx = torch.unique(node_idx)
            
            if node_idx.size(0) >= num_member:
                task[center] = node_idx[:num_member].tolist()
        
        return task


def patch_dataloader(dataset, centroid_nodes):
    """
    Patch the dataloader to use centroid nodes
    """
    # Store original classes and methods
    from data.dataloader import KGNeighborTask, NeighborTask, BatchSampler, ParamSampler

    # Original task classes
    OriginalKGNeighborTask = copy.deepcopy(KGNeighborTask)
    OriginalNeighborTask = copy.deepcopy(NeighborTask)

    # Patch sample method for KGNeighborTask
    def patched_kg_sample(self, num_label, num_member, num_shot, num_query, rng):
        # Use centroid nodes instead of random ones
        max_nodes = min(num_label, len(centroid_nodes))
        task = {}
        
        for i in range(max_nodes):
            center = centroid_nodes[i]
            node_idx = torch.ones(num_member * 10, dtype=torch.long) * center
            node_idx = self.neighbor_sampler.random_walk(node_idx, self.direction)
            node_idx = torch.unique(node_idx)
            
            # Sample edges around these nodes
            edge_idx = self.neighbor_sampler.sample_edge(node_idx, "inout")
            edge_idx = torch.unique(edge_idx)
            
            if edge_idx.size(0) >= num_member:
                task[center] = edge_idx[:num_member].tolist()
                if not self.is_multiway and len(task) == 1:
                    num_member = 1
        
        return task

    # Patch sample method for NeighborTask
    def patched_neighbor_sample(self, num_label, num_member, num_shot, num_query, rng):
        max_nodes = min(num_label, len(centroid_nodes))
        task = {}
        
        for i in range(max_nodes):
            center = centroid_nodes[i]
            node_idx = torch.ones(num_member * 10, dtype=torch.long) * center
            node_idx = self.neighbor_sampler.random_walk(node_idx, self.direction)
            node_idx = torch.unique(node_idx)
            
            if node_idx.size(0) >= num_member:
                task[center] = node_idx[:num_member].tolist()
        
        return task

    # Apply patched methods
    KGNeighborTask.sample = patched_kg_sample
    NeighborTask.sample = patched_neighbor_sample
    
    print("✅ Successfully patched dataloader to use centroid nodes")


def run_with_centroids():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run PG-NM with centroid-based node selection')
    parser.add_argument('--centroid-file', type=str, default=None, 
                        help='Path to file with centroid node IDs (one per line)')
    parser.add_argument('--use-mag-subset', type=bool, default=True,
                        help='Whether to use MAG240M subset')
    parser.add_argument('--mag-subset-size', type=int, default=20000000,
                        help='Size of the MAG240M subset')


    args, unknown = parser.parse_known_args()
    
    # Get standard PRODIGY parameters (this will parse the remaining args)
    params = get_params()
    
    # Ensure we're using neighbor_matching task
    if params['task_name'] != 'neighbor_matching':
        print(f"Warning: Task was set to {params['task_name']}, changing to neighbor_matching for PG-NM")
        params['task_name'] = 'neighbor_matching'
    
    # Load centroid nodes
    centroid_nodes = []
    dataset_path = params['root']
    dataset_name = params['dataset']
    
    # First try using command line arg
    if args.centroid_file and os.path.exists(args.centroid_file):
        centroid_file = args.centroid_file
    else:
        # Fall back to default location
        centroid_file = os.path.join(dataset_path, dataset_name, 'centroid_nodes.txt')
    
    if os.path.exists(centroid_file):
        print(f"Loading centroid nodes from {centroid_file}")
        with open(centroid_file, 'r') as f:
            centroid_nodes = [int(line.strip()) for line in f]
    else:
        # Check for PyTorch file
        centroid_pt_file = os.path.join(dataset_path, dataset_name, 'centroid_nodes.pt')
        if os.path.exists(centroid_pt_file):
            print(f"Loading centroid nodes from {centroid_pt_file}")
            data = torch.load(centroid_pt_file)
            centroid_nodes = data['centroid_nodes']
            optimal_k = data['optimal_k']
        else:
            print(f"⚠️ No centroid nodes found at {centroid_file} or {centroid_pt_file}")
            print("⚠️ Please run select_centroids.py first, or continuing with random node selection")
    

    dataset_args = {
        "root": params["root"],
        "dataset": params["dataset"],
        "force_cache": params["force_cache"],
        "small_dataset": params["small_dataset"],
        "invalidate_cache": None,
        "original_features": params["original_features"],
        "n_shot": params["n_shots"],
        "n_query": params["n_query"],
        "bert": None if params["original_features"] else params["bert_emb_model"],
        "bert_device": params["device"],
        "val_len_cap": params["val_len_cap"],
        "test_len_cap": params["test_len_cap"],
        "dataset_len_cap": params["dataset_len_cap"],
        "n_way": params["n_way"],
        "rel_sample_rand_seed": params["rel_sample_random_seed"],
        "calc_ranks": params["calc_ranks"],
        "kg_emb_model": params["kg_emb_model"] if params["kg_emb_model"] != "" else None,
        "task_name": params["task_name"],
        "shuffle_index": params["shuffle_index"],
        "node_graph": params["task_name"] == "sn_neighbor_matching"
    }
    
    # Add MAG240M specific parameters
    if params["dataset"] == "mag240m":
        dataset_args["use_subset"] = args.use_mag_subset
        dataset_args["subset_size"] = args.mag_subset_size
        print(f"Using MAG240M {'subset' if args.use_mag_subset else 'full dataset'}")
        if args.use_mag_subset:
            print(f"Subset size: {args.mag_subset_size} nodes")

    # Load the dataset
    datasets = get_dataset_wrap(**dataset_args)
    
    # Patch dataloader if we have centroid nodes
    if centroid_nodes:
        patch_dataloader(datasets, centroid_nodes)
        print(f"Using {len(centroid_nodes)} centroid nodes for training")
        
        # Optionally adjust the number of ways to match the number of centroids
        if params["n_way"] > len(centroid_nodes):
            print(f"Adjusting n_way from {params['n_way']} to {len(centroid_nodes)} to match centroids")
            params["n_way"] = len(centroid_nodes)
    
    # Create and run the trainer
    trainer = TrainerFS(datasets, params)
    trainer.train()


if __name__ == "__main__":
    run_with_centroids()