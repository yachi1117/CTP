import torch
import copy
from torch.utils.data import DataLoader
from data.dataloader import BatchSampler, ParamSampler, Collator
from data.augment import get_aug
from centroid_neighbor_task import CentroidNeighborTask

# Store a reference to the original function
from data.mag240m import get_mag240m_dataloader as original_get_mag240m_dataloader

def patched_get_mag240m_dataloader(dataset, task_name, split, node_split, batch_size, n_way, n_shot, n_query, 
                                  batch_count, root, num_workers, aug, aug_test, use_subset=True, subset_size=20000000, 
                                  **kwargs):
    """Patched version that supports centroid-based sampling"""
    
    # Check if centroids are available
    centroids = getattr(dataset, 'centroids', None)
    num_centroids = getattr(dataset, 'num_centroids', 10)
    
    if task_name == "neighbor_matching" and centroids is not None:
        print(f"Using centroid-based neighbor task sampling with {num_centroids} centroids per batch")
        seed = sum(ord(c) for c in split)
        if split == "train" or aug_test:
            aug = get_aug(aug, dataset.graph.x)
        else:
            aug = get_aug("")
        
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 2
        
        # Use custom CentroidNeighborTask
        task = CentroidNeighborTask(neighbor_sampler, len(dataset), "inout", centroids, num_centroids)
            
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = torch.zeros(1, 768).expand(len(dataset), -1)
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=Collator(label_meta, aug=aug))
        return dataloader
    else:
        # For other tasks or no centroids, use the original function
        return original_get_mag240m_dataloader(dataset, task_name, split, node_split, batch_size, n_way, 
                                              n_shot, n_query, batch_count, root, num_workers, aug, aug_test, 
                                              use_subset, subset_size, **kwargs)

# Replace the original function with our patched version
import data.mag240m
data.mag240m.get_mag240m_dataloader = patched_get_mag240m_dataloader