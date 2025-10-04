import os
import random
from collections import defaultdict
import copy
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from experiments.sampler import NeighborSamplerCacheAdj
from ogb.lsc import MAG240MDataset
from .dataset import SubgraphDataset
from .dataloader import NeighborTask, MultiTaskSplitWay, MultiTaskSplitBatch, MulticlassTask, ParamSampler, BatchSampler, Collator, ContrastiveTask
from .augment import get_aug
from .mag240m_subset import create_mag_subset_efficient

class MAG240MSubgraphDataset(SubgraphDataset):
    def get_subgraph(self, *args, **kwargs):
        graph = super().get_subgraph(*args, **kwargs)
        graph.x = graph.x.float()  # it was half
        return graph


def get_mag240m_dataset(root, n_hop=2, use_subset=True, subset_size=20000000, **kwargs):
    if use_subset:
        # Create or load the subset
        print(f"Using MAG240M subset with {subset_size} nodes")
        subset_graph = create_mag_subset_efficient(root, subset_size)
        
        # Create graph for neighbor sampler
        graph_ns = Data(edge_index=subset_graph.edge_index, num_nodes=subset_graph.num_nodes)
        
        # Setup neighbor sampler
        cache_path = os.path.join(root, "mag240m_subset", f"mag240m_subset_{subset_size}_adj.pt")
        neighbor_sampler = NeighborSamplerCacheAdj(cache_path, graph_ns, n_hop)
        
        return MAG240MSubgraphDataset(subset_graph, neighbor_sampler)
    else:
        # Original code for full MAG240M dataset
        dataset = MAG240MDataset(root)

        # Check if "mag240m_fts_adj_label.pt" exists, otherwise load
        # "mag240m_fts_adj.pt" and save it with labels.
        do_load = False
        adj_bi_cached = os.path.exists(os.path.join(root, "mag240m_adj_bi.pt"))
        if do_load and os.path.exists(os.path.join(root, "mag240m_fts_adj_label.pt")):
            print("Loading MAG240M dataset from .pt...")
            edge_index, fts, num_paper = torch.load(os.path.join(root, "mag240m_fts_adj_label.pt"))
        else:
            print("Loading MAG240M dataset...")
            t = time.time()
            edge_index = None
            if not adj_bi_cached:
                edge_index = dataset.edge_index('paper', 'cites', 'paper')
                edge_index = torch.from_numpy(edge_index)
            fts = torch.from_numpy(dataset.paper_feat)
            num_paper = dataset.num_papers
            data = (edge_index, fts, dataset.num_papers)
            print(f"Loading MAG240M dataset takes {time.time() - t:.2f} seconds")
            if do_load:
                torch.save(data, os.path.join(root, "mag240m_fts_adj_label.pt"))
        print("Done loading MAG240M dataset.")
        graph_ns = None if adj_bi_cached else Data(edge_index=edge_index, num_nodes=num_paper)
        neighbor_sampler = NeighborSamplerCacheAdj(os.path.join(root, "mag240m_adj_bi.pt"), graph_ns, n_hop)
        print("Done loading MAG240M neighbor sampler.")
        graph = Data(x=fts, num_nodes=num_paper)
        return MAG240MSubgraphDataset(graph, neighbor_sampler)


def mag240m_labels(split, node_split="", root="dataset", remove_cs=True, use_subset=True, subset_size=20000000):
    dataset = MAG240MDataset(root)
    num_classes = dataset.num_classes
    
    # Load the subset node mapping if using subset
    if use_subset:
        subset_nodes_path = os.path.join(root, "mag240m_subset", f"mag240m_subset_{subset_size}_nodes.pt")
        if os.path.exists(subset_nodes_path):
            subset_nodes = torch.load(subset_nodes_path)
        else:
            # If we haven't created the mapping yet (unlikely if dataset exists)
            print("Warning: Subset nodes mapping not found. This should have been created with the subset.")
            subset_nodes = torch.arange(subset_size)
            
    if remove_cs:
        arxiv_labels = [
            0, 1, 3, 6, 9, 16, 17, 23, 24, 26,
            29, 39, 42, 47, 52, 57, 59, 63, 73, 77,
            79, 85, 86, 89, 94, 95, 105, 109, 114, 119,
            120, 122, 124, 130, 135, 137, 139, 147, 149, 152]
        labels = list(set(range(num_classes)) - set(arxiv_labels))
        additional = arxiv_labels
    else:
        labels = list(range(num_classes))
        additional = []
    generator = random.Random(42)
    generator.shuffle(labels)

    test_val_length = 5
    TEST_LABELS = labels[:test_val_length] + additional
    VAL_LABELS = labels[test_val_length: test_val_length * 2] + additional
    TRAIN_LABELS = labels[test_val_length * 2:]

    if use_subset:
        # Get labels for subset nodes only
        label = dataset.paper_label[subset_nodes]
    else:
        # Original code for full dataset
        label = dataset.all_paper_label
        
    if split == "train":
        label_set = set(TRAIN_LABELS)
    elif split == "val":
        label_set = set(VAL_LABELS)
    elif split == "test":
        if not remove_cs:
            print("Warning: remove_cs is set to false, might not be enough samples.")
        label_set = set(TEST_LABELS)
    else:
        raise ValueError(f"Invalid split: {split}")

    return label, label_set, num_classes


def get_mag240m_dataloader(dataset, task_name, split, node_split, batch_size, n_way, n_shot, n_query, batch_count, root, num_workers, aug, aug_test, use_subset=True, subset_size=20000000, **kwargs):
    seed = sum(ord(c) for c in split)
    if split == "train" or aug_test:
        aug = get_aug(aug, dataset.graph.x)
    else:
        aug = get_aug("")
    if task_name == "same_graph":
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 0
        sampler = BatchSampler(
            batch_count,
            ContrastiveTask(len(dataset)),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = torch.zeros(1, 768).expand(len(dataset), -1)
    elif task_name == "neighbor_matching":
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 2
        sampler = BatchSampler(
            batch_count,
            NeighborTask(neighbor_sampler, len(dataset), "inout"),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = torch.zeros(1, 768).expand(len(dataset), -1)
    elif task_name == "classification":
        labels, label_set, num_classes = mag240m_labels(split, node_split, root=root, 
                                                     remove_cs=True, use_subset=use_subset, 
                                                     subset_size=subset_size)
        sampler = BatchSampler(
            batch_count,
            MulticlassTask(labels, label_set),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = torch.zeros(1, 768).expand(num_classes, -1)
    # Classification and neighbor matching - multitask splitbatch
    elif task_name.startswith("cls_nm"):
        labels, label_set, num_classes = mag240m_labels(split, node_split, root=root, 
                                                     use_subset=use_subset, subset_size=subset_size)
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 2
        
        if task_name.endswith("sb"):
            task_base = MultiTaskSplitBatch([
                MulticlassTask(labels, label_set),
                NeighborTask(neighbor_sampler, len(dataset), "inout")
            ], ["mct", "nt"], [1, 3])
        elif task_name.endswith("sw"):
            task_base = MultiTaskSplitWay([
                MulticlassTask(labels, label_set), 
                NeighborTask(neighbor_sampler, len(dataset), "inout")
            ], ["mct", "nt"], split="even")
        
        sampler = BatchSampler(
            batch_count,
            task_base,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_meta = {}
        label_meta["mct"] = torch.zeros(1, 768).expand(num_classes, -1)
        label_meta["nt"] = torch.zeros(1, 768).expand(len(dataset), -1)
    else:
        raise ValueError(f"Unknown task for MAG240M: {task_name}")
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=Collator(label_meta, aug=aug))
    return dataloader


if __name__ == "__main__":
    from tqdm import tqdm
    import cProfile

    root = "../FSdatasets/mag240m"
    n_hop = 2

    dataset = get_mag240m_dataset(root, n_hop, use_subset=True, subset_size=20000000)
    dataloader = get_mag240m_dataloader(dataset, "classification", "train", "", 5, 3, 3, 24, 10000, root, 10, 
                                      aug="", aug_test=False, use_subset=True, subset_size=20000000)

    for batch in tqdm(dataloader):
        pass