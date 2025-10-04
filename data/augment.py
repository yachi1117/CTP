import random
import copy
import torch


class AugBase:
    def __call__(self, graph):
        raise NotImplementedError


class Compose(AugBase):
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, graph):
        for aug in self.augs:
            graph = aug(graph)
        return graph


class Identity(AugBase):
    def __call__(self, graph):
        return graph


class DropNode(AugBase):
    def __init__(self, drop_percent=0.3):
        self.drop_percent = drop_percent

    def __call__(self, graph):
        num_node = graph.num_nodes
        num_drop = int(num_node * self.drop_percent)
        node_drop = random.sample(range(num_node), num_drop)
        node_mask = torch.ones(num_node, dtype=bool)
        node_mask[node_drop] = False
        node_mask[0] = True  # center node
        node_mask[-1] = True  # super node

        graph = copy.copy(graph)
        edge_index = graph.edge_index
        edge_mask = (node_mask[edge_index[0]]).logical_and(node_mask[edge_index[1]])
        edge_index = edge_index[:, edge_mask]
        if "edge_attr" in graph and graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr[edge_mask]
        graph.edge_index = edge_index
        graph.node_mask = node_mask
        return graph


class ZeroNodeAttr(AugBase):
    def __init__(self, mask_percent=0.3):
        self.mask_percent = mask_percent

    def __call__(self, graph):
        num_node = graph.num_nodes
        num_drop = int(num_node * self.mask_percent)
        node_drop = random.sample(range(num_node), num_drop)
        node_mask = torch.ones(num_node, dtype=bool)
        node_mask[node_drop] = False

        graph = copy.copy(graph)
        graph.x_orig = graph.x
        graph.x = graph.x * node_mask.unsqueeze(1)
        if hasattr(graph, "node_attr_mask"):
            graph.node_attr_mask = graph.node_attr_mask.logical_and(node_mask)
        else:
            graph.node_attr_mask = node_mask
        return graph


class RandomNodeAttr(AugBase):
    def __init__(self, distribution, mask_percent=0.3):
        self.distribution = distribution
        self.mask_percent = mask_percent

    def __call__(self, graph):
        num_node = graph.num_nodes
        num_drop = int(num_node * self.mask_percent)
        node_drop = random.sample(range(num_node), num_drop)
        node_mask = torch.ones(num_node, dtype=bool)
        node_mask[node_drop] = False

        graph = copy.copy(graph)
        graph.x_orig = graph.x
        graph.x = graph.x.clone()
        random_idx = random.sample(range(self.distribution.size(0)), len(node_drop))
        graph.x[node_drop] = self.distribution[random_idx].float()
        if hasattr(graph, "node_attr_mask"):
            graph.node_attr_mask = graph.node_attr_mask.logical_and(node_mask)
        else:
            graph.node_attr_mask = node_mask
        return graph

class ProtectedDropNode(AugBase):
    """
    Augmentation that protects a percentage of nodes globally and then applies node dropping
    to the remaining nodes.
    
    Args:
        drop_percent (float): Percentage of non-protected nodes to drop
        protect_percent (float): Percentage of nodes to protect from dropping
        seed (int, optional): Random seed for determining which nodes to protect
    """
    def __init__(self, drop_percent=0.3, protect_percent=0.2, seed=None):
        self.drop_percent = drop_percent
        self.protect_percent = protect_percent
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random
        
    def __call__(self, graph):
        num_node = graph.num_nodes
        
        # Generate protection mask
        eligible_nodes = list(range(2, num_node - 1)) if num_node > 3 else []
        num_protect = min(int(num_node * self.protect_percent), len(eligible_nodes))
        
        # Use seed-controlled RNG for protection mask
        protect_indices = self.rng.sample(eligible_nodes, num_protect) if num_protect > 0 else []
        protect_mask = torch.zeros(num_node, dtype=bool)
        protect_mask[protect_indices] = True
        
        # Always protect center and super nodes
        if num_node > 0:
            protect_mask[0] = True  # center node
        if num_node > 1:
            protect_mask[-1] = True  # super node
        
        # Calculate number of nodes to drop from non-protected nodes
        non_protected_indices = [i for i in range(num_node) if not protect_mask[i]]
        non_protected_count = len(non_protected_indices)
        num_drop = min(int(non_protected_count * self.drop_percent), non_protected_count)
        
        # Using the global random for drop decisions (as per original implementation)
        # Could also use self.rng for this if desired
        drop_indices = random.sample(non_protected_indices, num_drop) if num_drop > 0 else []
        
        node_mask = torch.ones(num_node, dtype=bool)
        node_mask[drop_indices] = False
        
        graph = copy.copy(graph)
        edge_index = graph.edge_index
        edge_mask = (node_mask[edge_index[0]]).logical_and(node_mask[edge_index[1]])
        edge_index = edge_index[:, edge_mask]
        
        if "edge_attr" in graph and graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr[edge_mask]
            
        graph.edge_index = edge_index
        graph.node_mask = node_mask
        graph.protect_mask = protect_mask  # Store the protection mask for reference
        return graph

def get_aug(aug_spec, node_feature_distribution=None):
    if not aug_spec:
        return Identity()
    augs = []
    for spec in aug_spec.split(","):
        if spec.startswith("ND"):
            augs.append(DropNode(float(spec[2:])))
        elif spec.startswith("NZ"):
            augs.append(ZeroNodeAttr(float(spec[2:])))
        elif spec.startswith("NR"):
            if node_feature_distribution is None:
                raise ValueError(f"node_feature_distribution not defined for RandomNodeAttr")
            augs.append(RandomNodeAttr(node_feature_distribution, float(spec[2:])))
        elif spec.startswith("GP"):
            # Format: GP<drop_percent>P<protect_percent>S<seed>
            # Example: GP0.3P0.2S42 = drop 30% of nodes, protect 20% with seed 42
            parts = spec[2:].split('P')
            drop_pct = 0.3  # Default
            protect_pct = 0.2  # Default
            seed = None  # Default: no seed
            
            # Parse drop percentage
            if len(parts) > 0 and parts[0]:
                drop_pct = float(parts[0])
            
            # Parse protection percentage and seed
            if len(parts) > 1:
                protect_parts = parts[1].split('S')
                if protect_parts[0]:
                    protect_pct = float(protect_parts[0])
                
                # Parse seed if present
                if len(protect_parts) > 1 and protect_parts[1]:
                    seed = int(protect_parts[1])
            
            augs.append(ProtectedDropNode(drop_pct, protect_pct, seed))
        else:
            raise ValueError(f"Unknown augmentation {spec}")
    return Compose(augs)