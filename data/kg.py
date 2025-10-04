import os
from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib
from data.load_kg_dataset import SubgraphFewshotDatasetWithTextFeats
from .dataset import KGSubgraphDataset
from .dataloader import (
    KGNeighborTask,
    MulticlassTask,
    ParamSampler,
    BatchSampler,
    KGCollator,
    ContrastiveTask,
    NeighborTask,
    Collator,
    MultiTaskSplitBatch,
    MultiTaskSplitWay,
    TaskBase  # Import TaskBase if needed, though FunctionTaskAdapter doesn't inherit
)
from .augment import get_aug
from torch_geometric.data import Data
from experiments.sampler import NeighborSamplerCacheAdj
import copy
import random
import numpy as np
import json

class FunctionTaskAdapter(TaskBase): # Inherit from TaskBase for consistency if desired
    """Adapter class to wrap a function so it can be used as a task in BatchSampler"""
    def __init__(self, sample_fn):
        self.sample_fn = sample_fn

    # Ensure this sample method accepts all arguments from BatchSampler
    def sample(self, num_label, num_member, num_shot, num_query, rng, batch_idx=None):
        # Pass all received arguments to the wrapped function
        return self.sample_fn(num_label, num_member, num_shot, num_query, rng, batch_idx)


def get_csr_split(root, name):
    # get CSR label split for the given dataset
    result = {}
    for subset in ["pretrain", "dev", "test"]:
        fname = subset + "_tasks"
        fname += ".json"
        fname = os.path.join(root, name, fname)
        if subset not in result:
            result[subset] = set()
        if os.path.exists(fname):
            #with open(fname) as f:
            #    result[subset] = set(json.load(f)).union(result[subset])
            result[subset] = set(list(json.load(open(fname)).keys())).union(result[subset])
    return result


def get_kg_dataset(root, name, n_hop=2, bert=None, bert_device="cpu", **kwargs):
    assert name in ["NELL", "FB15K-237", "ConceptNet", "Wiki", "WikiKG90M"]

    kind = "union"
    sampler_type = "new"
    subset = "test" # This seems fixed, might need adjustment if different subsets are used elsewhere
    hop = 2
    shot = 3 # This seems fixed, might need adjustment

    if name == "ConceptNet":
        hop = 1
    if name == "NELL":
        hop = 1
    if name == "FB15K-237":
        hop = 1
    pretrained_embeddings = None
    dataset = SubgraphFewshotDatasetWithTextFeats(root=root, dataset=name, mode=subset, hop=hop, kind = kind, shot=shot, preprocess=False,
                     bert=bert, device=bert_device, embeddings_model=pretrained_embeddings, graph_only = True)

    graph_ns = Data(edge_index=dataset.graph.edge_index, num_nodes=dataset.graph.num_nodes)
    adj_path = os.path.join(root, name, f"{name}_adj.pt")
    print(f"Loading adjacent matrix for neighbor sampling from {adj_path}")
    neighbor_sampler = NeighborSamplerCacheAdj(adj_path, graph_ns, hop)
    print(f"Loaded adjacent matrix for neighbor sampling from {adj_path}")
    dataset.csr_split = get_csr_split(root, name)
    node_graph_flag = kwargs.get("node_graph", False) # Use .get for safety
    return KGSubgraphDataset(dataset, neighbor_sampler, sampler_type, node_graph = node_graph_flag)


def idx_split(n, fracs=[0.7, 0.1, 0.2]):
    generator = random.Random(42)
    labels = list(range(n))
    generator.shuffle(labels)
    i = int(n * fracs[0])
    j = int(n * (fracs[0] + fracs[1]))
    train = labels[:i]
    val = labels[i:j]
    test = labels[j:]
    return {"train": train, "valid": val, "test": test}


def kg_labels(dataset, split, node_split = "", all_test=False, csr_split=False):
    num_classes = dataset.pyg_graph.edge_attr.max().item() +1
    print("Number of classes (relations):", num_classes)
    labels = list(range(num_classes))
    generator = random.Random(42)
    generator.shuffle(labels)

    if csr_split:
        print("Using CSR split for labels...")
        train_tasks, test_tasks, val_tasks = dataset.kg_dataset.csr_split["pretrain"], dataset.kg_dataset.csr_split["test"], dataset.kg_dataset.csr_split["dev"]
        assert train_tasks.intersection(test_tasks) == set() and train_tasks.intersection(val_tasks) == set() and test_tasks.intersection(val_tasks) == set()
        # Ensure label_text exists and is populated correctly
        if not hasattr(dataset, 'label_text') or not dataset.label_text:
             raise ValueError("dataset.label_text is missing or empty, required for CSR split mapping.")
        TRAIN_LABELS = [dataset.label_text.index(task) for task in train_tasks if task in dataset.label_text]
        VAL_LABELS = [dataset.label_text.index(task) for task in val_tasks  if task in dataset.label_text]
        TEST_LABELS = [dataset.label_text.index(task) for task in test_tasks  if task in dataset.label_text]
        print(f"CSR Split - Train: {len(TRAIN_LABELS)}, Val: {len(VAL_LABELS)}, Test: {len(TEST_LABELS)}")
    elif all_test:
        TEST_LABELS = labels
        VAL_LABELS = labels
        TRAIN_LABELS = labels
        print("Setting all labels for evaluation...")
    else:
        print("Using default percentage split for labels...")
        if num_classes <= 20:
            # ConceptNet
            i = int(num_classes / 3)
            j = int(num_classes * 2/3)
        else:
            # FB and NELL, Wiki etc.
            i = int(num_classes * 0.6)
            j = int(num_classes * 0.8)

        TEST_LABELS = labels[:i]
        VAL_LABELS = labels[i: j]
        TRAIN_LABELS = labels[j:]
        print("TEST_LABELS", len(TEST_LABELS))
        print("VAL_LABELS", len(VAL_LABELS))
        print("TRAIN_LABELS", len(TRAIN_LABELS))

    label_attr = dataset.pyg_graph.edge_attr # This should be the edge attribute tensor
    if split == "train":
        label_set = set(TRAIN_LABELS)
    elif split == "val":
        label_set = set(VAL_LABELS)
    elif split == "test":
        label_set = set(TEST_LABELS)
    else:
        raise ValueError(f"Invalid split: {split}")

    return label_attr, label_set, num_classes

def kg_task_no_labels_split(labels, dataset, label_set, linear_probe, train_cap=3, split="train"):
    # labels = edge_attr
    edge_index = dataset.pyg_graph.edge_index
    # Ensure edge_index and labels have compatible lengths if labels correspond to edges
    if len(labels) != edge_index.shape[1]:
         print(f"Warning: Length mismatch between labels ({len(labels)}) and edge_index columns ({edge_index.shape[1]}). Assuming labels correspond to edges.")
         # Decide how to handle mismatch - truncate labels? error? For now, assume it's okay.

    # Use number of edges for splitting if labels correspond to edges
    num_items_to_split = edge_index.shape[1]
    rnd_split = idx_split(num_items_to_split) # Split indices from 0 to num_edges-1

    train_label = labels.numpy().copy() # Make a copy to modify

    # Apply split based on edge indices
    train_split_indices = np.array(rnd_split["train"])
    val_split_indices = np.array(rnd_split["valid"])
    test_split_indices = np.array(rnd_split["test"])

    # Masking logic: negative values indicate masked labels
    masked_value_offset = labels.max() + 1 # Ensure masked values don't clash with real labels

    current_split_indices = None
    if split == "train":
        current_split_indices = train_split_indices
        # Mask everything *not* in the train split
        mask_out_indices = np.concatenate((val_split_indices, test_split_indices))
        train_label[mask_out_indices] = -masked_value_offset - train_label[mask_out_indices] # Mask non-train labels
    elif split == "val":
        current_split_indices = val_split_indices
        # Mask everything *not* in the val split
        mask_out_indices = np.concatenate((train_split_indices, test_split_indices))
        train_label[mask_out_indices] = -masked_value_offset - train_label[mask_out_indices] # Mask non-val labels
    elif split == "test":
        current_split_indices = test_split_indices
         # Mask everything *not* in the test split
        mask_out_indices = np.concatenate((train_split_indices, val_split_indices))
        train_label[mask_out_indices] = -masked_value_offset - train_label[mask_out_indices] # Mask non-test labels

    # Apply training cap if specified (only affects the train split conceptually)
    train_label_for_cap = labels.numpy().copy() # Original labels for capping counts
    if split == "train" and train_cap is not None:
        for i in range(labels.max().item() + 1):
            # Find indices in the current split that have label i
            label_indices_in_split = current_split_indices[train_label_for_cap[current_split_indices] == i]
            if len(label_indices_in_split) > train_cap:
                # Indices to disable (mask) are those beyond the cap
                disabled_indices = label_indices_in_split[train_cap:]
                # Apply masking to the `train_label` array
                train_label[disabled_indices] = -masked_value_offset - train_label[disabled_indices]

    # Final label array for the task: unmasked values are positive, masked are negative
    label = train_label

    # `train_label` argument for MulticlassTask seems intended for linear probing separation
    # For non-linear probe, it's usually None. Let's stick to that unless linear_probe is True.
    train_label_arg = None
    if linear_probe:
         # If linear probing, provide the labels *only* for the training split indices
         # This needs careful implementation based on how linear probing uses it.
         # A common pattern is to use a separate set of nodes/edges for training the probe head.
         # The current implementation doesn't seem fully set up for this.
         # Setting to None for now to avoid potential issues.
         print("Warning: Linear probe train_label separation in kg_task_no_labels_split might need review.")
         pass # train_label_arg remains None

    # Ensure label_set is usable by MulticlassTask
    if isinstance(label_set, set):
        label_set = list(label_set)

    return MulticlassTask(label, label_set, train_label=train_label_arg, linear_probe=linear_probe)


def get_kg_dataloader(dataset, task_name, split, node_split, batch_size, n_way, n_shot, n_query, batch_count, root, num_workers, aug, aug_test, train_cap, linear_probe, label_set=None, all_test=False, **kwargs):
    seed = sum(ord(c) for c in split + task_name) # Simple seed based on split and task
    #seed = None # Uncomment for non-deterministic sampling within an epoch

    split_labels = kwargs.get("split_labels", True) # Use .get for safety
    csr_split = kwargs.get("csr_split", False)

    if split == "train" or aug_test:
        aug_obj = get_aug(aug, dataset.pyg_graph.x) # Pass features for potential feature-based aug
    else:
        aug_obj = get_aug("") # No augmentation for val/test unless aug_test is True

    is_multiway = True
    effective_n_way = n_way
    if n_way == 1:
        # Binary classification case (e.g., link prediction style)
        # n_member becomes n_shot (positive) + n_query (negative)? Needs clarification.
        # The dataloader/collator logic handles the binary case based on is_multiway=False
        is_multiway = False
        effective_n_way = 2 # Conceptually 2-way (pos/neg), though sampling might differ

    task = None
    sampler = None
    label_meta = None # Will hold label embeddings or placeholders

    # --- Task & Sampler Setup ---
    if task_name == "same_graph":
        # Contrastive task based on node identity (sample node, return itself multiple times)
        num_nodes = dataset.pyg_graph.num_nodes
        task = ContrastiveTask(num_nodes)
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1), # n_aug=1 assumed
            seed=seed,
        )
        # Label meta might be node features or zeros if not used
        label_meta = torch.zeros(1, dataset.pyg_graph.x.shape[1]).expand(num_nodes, -1) # Use input dim

    elif task_name == "neighbor_matching":
        num_nodes = dataset.pyg_graph.num_nodes
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 1 # Usually 1-hop neighbors for this task

        # Use the specific KGNeighborTask
        kg_task = KGNeighborTask(dataset, neighbor_sampler, num_nodes, "inout", is_multiway)

        # Wrap KGNeighborTask.sample in an adapter function IF its signature doesn't match BatchSampler's call
        # BatchSampler calls task.sample(n_way, n_member, n_shot, n_query, rng, batch_idx)
        # KGNeighborTask.sample expects (self, num_label, num_member, num_shot, num_query, rng, batch_idx=None) - it should now match!
        # Therefore, adapter might not be strictly needed if KGNeighborTask is updated correctly.
        # Using FunctionTaskAdapter for robustness in case signatures diverge later.
        def sample_adapter(num_label, num_member, num_shot, num_query, rng, batch_idx=None):
             # Pass the arguments KGNeighborTask.sample expects
             return kg_task.sample(num_label, num_member, num_shot, num_query, rng, batch_idx)

        task = FunctionTaskAdapter(sample_adapter) # Use the adapter

        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1), # n_aug=1 assumed
            seed=seed,
        )
        # Label meta represents nodes in this case (e.g., node features or zeros)
        label_meta = torch.zeros(1, dataset.pyg_graph.x.shape[1]).expand(num_nodes, -1) # Use input dim

    elif task_name == "sn_neighbor_matching": # Standard node neighbor matching (not KG edge-centric)
        num_nodes = dataset.pyg_graph.num_nodes
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 1
        task = NeighborTask(neighbor_sampler, num_nodes, "inout") # Use standard NeighborTask
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        # Label meta represents nodes
        label_meta = torch.zeros(1, dataset.pyg_graph.x.shape[1]).expand(num_nodes, -1) # Use input dim
        # Note: This path uses the standard Collator, not KGCollator
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                                collate_fn=Collator(label_meta, aug=aug_obj, is_multiway=is_multiway))
        return dataloader # Return early as it uses a different collator

    elif task_name == "multiway_classification":
        labels, label_set_split, num_classes = kg_labels(dataset, split, node_split, all_test, csr_split)
        if split_labels:
            # Use the label set derived from the split (TRAIN_LABELS, VAL_LABELS, etc.)
            task = MulticlassTask(labels, label_set_split, train_label=None, linear_probe=linear_probe)
        else:
            # Use the provided label_set (or fail if None) and handle masking internally
            assert label_set is not None, "label_set must be provided for no_split_labels"
            task = kg_task_no_labels_split(labels, dataset=dataset, train_cap=train_cap, split=split,
                                           label_set=label_set, linear_probe=linear_probe)
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        # Label meta should be the actual relation embeddings
        if not hasattr(dataset, 'label_embeddings') or dataset.label_embeddings is None:
             raise ValueError("dataset.label_embeddings is required for multiway_classification")
        # Ensure label embeddings have the correct size (num_classes x embedding_dim)
        if dataset.label_embeddings.shape[0] != num_classes:
             print(f"Warning: Mismatch between num_classes ({num_classes}) and label_embeddings shape ({dataset.label_embeddings.shape}). Using available embeddings.")
             # Decide how to handle: error out, pad, or use as is? Using as is for now.
        label_meta = torch.clone(dataset.label_embeddings)

    elif task_name == "cls_nm": # Combined classification and neighbor matching
        # 1. Classification Part
        labels_cls, label_set_cls, num_classes = kg_labels(dataset, split, node_split, all_test, csr_split)
        if split_labels:
            task_cls = MulticlassTask(labels_cls, label_set_cls, train_label=None, linear_probe=linear_probe)
        else:
             assert label_set is not None, "label_set must be provided for no_split_labels"
             task_cls = kg_task_no_labels_split(labels_cls, dataset=dataset, train_cap=train_cap, split=split,
                                                label_set=label_set, linear_probe=linear_probe)

        # 2. Neighbor Matching Part
        num_nodes = dataset.pyg_graph.num_nodes
        neighbor_sampler = copy.copy(dataset.neighbor_sampler)
        neighbor_sampler.num_hops = 1
        kg_task_nm = KGNeighborTask(dataset, neighbor_sampler, num_nodes, "inout", is_multiway)

        # Adapter for neighbor matching part (similar to neighbor_matching task)
        def sample_adapter_nm(num_label, num_member, num_shot, num_query, rng, batch_idx=None):
             return kg_task_nm.sample(num_label, num_member, num_shot, num_query, rng, batch_idx)
        task_adapter_nm = FunctionTaskAdapter(sample_adapter_nm)

        # Combine tasks
        task_names = ["mct", "nt"] # MultiClassTask, NeighborTask(KG adapted)
        if "sw" in task_name: # Split ways between tasks
            task_base = MultiTaskSplitWay([task_cls, task_adapter_nm], task_names, split="even")
        else: # Split batches between tasks (e.g., 98% MCT, 2% NT)
            # Make task counts configurable if needed
            task_base = MultiTaskSplitBatch([task_cls, task_adapter_nm], task_names, [98, 2])

        sampler = BatchSampler(
            batch_count,
            task_base,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )

        # Label meta needs entries for both tasks
        label_meta = {}
        # Classification label embeddings
        if not hasattr(dataset, 'label_embeddings') or dataset.label_embeddings is None:
             raise ValueError("dataset.label_embeddings is required for cls_nm task")
        if dataset.label_embeddings.shape[0] != num_classes:
             print(f"Warning: Mismatch between num_classes ({num_classes}) and label_embeddings shape ({dataset.label_embeddings.shape}) for cls_nm MCT.")
        label_meta["mct"] = torch.clone(dataset.label_embeddings)
        # Neighbor matching node placeholders
        label_meta["nt"] = torch.zeros(1, dataset.pyg_graph.x.shape[1]).expand(num_nodes, -1)

    else:
        raise ValueError(f"Unknown task for KG: {task_name}")

    # --- Dataloader Construction ---
    # Use KGCollator for KG tasks, handles edge-centric sampling details
    collator = KGCollator(label_meta, aug=aug_obj, is_multiway=is_multiway)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collator)

    return dataloader


if __name__ == "__main__":
    from tqdm import tqdm
    import cProfile

    root = "../FSdatasets/mag240m"
    n_hop = 2

    dataset = get_mag240m_dataset(root, n_hop)
    dataloader = get_mag240m_dataloader(dataset, "train", "", 5, 3, 3, 24, 10000, root, 10)

    for batch in tqdm(dataloader):
        pass