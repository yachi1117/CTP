import random
import copy
import torch
from .augment import AugBase, DropNode, ZeroNodeAttr, RandomNodeAttr, Compose, Identity, get_aug


# New augmentation classes that maintain global state
class GlobalDropNode(AugBase):
    """Drop nodes consistently across different subgraphs."""
    def __init__(self, drop_percent=0.3, seed=42):
        self.drop_percent = drop_percent
        self.rng = random.Random(seed)
        self.dropped_nodes = {}  # Map from node ID to whether it's dropped
        
    def __call__(self, graph):
        num_node = graph.num_nodes
        
        # Get the node IDs
        if hasattr(graph, 'x_id'):
            node_ids = graph.x_id.cpu().numpy().tolist()
        else:
            # If graph doesn't have x_id, use range(num_node)
            node_ids = list(range(num_node))
        
        # Initialize a mask for this graph
        node_mask = torch.ones(num_node, dtype=bool)
        
        # For each node, check if it's already in our global buffer
        # If not, decide whether to drop it
        for i, node_id in enumerate(node_ids):
            if node_id not in self.dropped_nodes:
                # Make a decision about this node
                self.dropped_nodes[node_id] = self.rng.random() < self.drop_percent
            
            # Apply the decision
            if self.dropped_nodes[node_id]:
                node_mask[i] = False
        
        # Ensure the center node and super node are not dropped
        node_mask[0] = True  # center node
        node_mask[-1] = True  # super node
        
        # Apply the mask to the graph
        graph = copy.copy(graph)
        edge_index = graph.edge_index
        edge_mask = (node_mask[edge_index[0]]).logical_and(node_mask[edge_index[1]])
        edge_index = edge_index[:, edge_mask]
        if "edge_attr" in graph and graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr[edge_mask]
        graph.edge_index = edge_index
        graph.node_mask = node_mask
        
        return graph

class GlobalZeroNodeAttr(AugBase):
    """Mask node attributes consistently across different subgraphs."""
    def __init__(self, mask_percent=0.3, seed=42):
        self.mask_percent = mask_percent
        self.rng = random.Random(seed)
        self.masked_nodes = {}  # Map from node ID to whether it's masked
        
    def __call__(self, graph):
        num_node = graph.num_nodes
        
        # Get the node IDs
        if hasattr(graph, 'x_id'):
            node_ids = graph.x_id.cpu().numpy().tolist()
        else:
            # If graph doesn't have x_id, use range(num_node)
            node_ids = list(range(num_node))
        
        # Initialize a mask for this graph
        node_mask = torch.ones(num_node, dtype=bool)
        
        # For each node, check if it's already in our global buffer
        # If not, decide whether to mask it
        for i, node_id in enumerate(node_ids):
            if node_id not in self.masked_nodes:
                # Make a decision about this node
                self.masked_nodes[node_id] = self.rng.random() < self.mask_percent
            
            # Apply the decision
            if self.masked_nodes[node_id]:
                node_mask[i] = False
        
        # Apply the mask to the graph
        graph = copy.copy(graph)
        graph.x_orig = graph.x.clone()
        graph.x = graph.x * node_mask.unsqueeze(1)
        if hasattr(graph, "node_attr_mask"):
            graph.node_attr_mask = graph.node_attr_mask.logical_and(node_mask)
        else:
            graph.node_attr_mask = node_mask
        
        return graph

class GlobalRandomNodeAttr(AugBase):
    """Replace node attributes with random values consistently across different subgraphs."""
    def __init__(self, distribution, mask_percent=0.3, seed=42):
        self.distribution = distribution
        self.mask_percent = mask_percent
        self.rng = random.Random(seed)
        self.masked_nodes = {}  # Map from node ID to whether it's masked
        self.random_idx = {}  # Map from node ID to random index in distribution
        
    def __call__(self, graph):
        num_node = graph.num_nodes
        
        # Get the node IDs
        if hasattr(graph, 'x_id'):
            node_ids = graph.x_id.cpu().numpy().tolist()
        else:
            # If graph doesn't have x_id, use range(num_node)
            node_ids = list(range(num_node))
        
        # Initialize a mask for this graph
        node_mask = torch.ones(num_node, dtype=bool)
        
        # For each node, check if it's already in our global buffer
        # If not, decide whether to mask it
        node_drop = []
        for i, node_id in enumerate(node_ids):
            if node_id not in self.masked_nodes:
                # Make a decision about this node
                self.masked_nodes[node_id] = self.rng.random() < self.mask_percent
                self.random_idx[node_id] = self.rng.randint(0, self.distribution.size(0) - 1)
            
            # Apply the decision
            if self.masked_nodes[node_id]:
                node_mask[i] = False
                node_drop.append(i)
        
        # Apply the mask to the graph
        graph = copy.copy(graph)
        graph.x_orig = graph.x.clone()
        graph.x = graph.x.clone()
        if node_drop:
            random_idx = [self.random_idx[node_ids[i]] for i in node_drop]
            graph.x[node_drop] = self.distribution[random_idx].float()
        if hasattr(graph, "node_attr_mask"):
            graph.node_attr_mask = graph.node_attr_mask.logical_and(node_mask)
        else:
            graph.node_attr_mask = node_mask
        
        return graph

class TwoPartAugmentation(AugBase):
    """
    Implementation of the two-part strategy:
    1. For a percentage of subgraphs, use the original intact structure
    2. For the remaining subgraphs, apply global augmentation
    """
    def __init__(self, aug, original_percent=0.5, seed=42):
        self.aug = aug
        self.original_percent = original_percent
        self.rng = random.Random(seed)
        self.graph_decisions = {}  # Map from graph ID to whether it's augmented
        
    def __call__(self, graph):
        # Use a hash of the graph's structure to identify it consistently
        graph_id = id(graph)
        
        # Make a copy of the graph to avoid modifying the original
        graph_copy = copy.copy(graph)
        
        # Check if we've made a decision about this graph before
        if graph_id not in self.graph_decisions:
            # Make a decision about this graph
            self.graph_decisions[graph_id] = self.rng.random() >= self.original_percent
        
        # If we've decided to augment this graph, apply the augmentation
        if self.graph_decisions[graph_id]:
            return self.aug(graph_copy)
        else:
            # For graphs we don't augment, we still need to add the same attributes
            # that would be added by the augmentation, but without modifying the data
            
            # Add node_mask attribute (used by GlobalDropNode)
            if not hasattr(graph_copy, "node_mask"):
                num_nodes = graph_copy.num_nodes
                graph_copy.node_mask = torch.ones(num_nodes, dtype=bool)
                
            # Add node_attr_mask attribute (used by GlobalZeroNodeAttr)
            if not hasattr(graph_copy, "node_attr_mask"):
                num_nodes = graph_copy.num_nodes
                graph_copy.node_attr_mask = torch.ones(num_nodes, dtype=bool)
                
            # Add x_orig attribute (used by both zero and random attr augmentations)
            if not hasattr(graph_copy, "x_orig") and hasattr(graph_copy, "x"):
                graph_copy.x_orig = graph_copy.x.clone()
                
            return graph_copy





# Modified get_aug function to support the new augmentation classes
def get_two_part_aug(aug_spec, node_feature_distribution=None, seed=42):
    """
    Get two-part global augmentation from specification.
    
    Special prefixes:
    - TP: Two-part augmentation (50% original, 50% augmented)
    - G: Global augmentation that maintains consistency across subgraphs
    
    Examples:
    - "TP:GND0.3,GNZ0.3": 50% original, 50% with global DropNode and ZeroNodeAttr
    - "GND0.3,GNZ0.3": All with global DropNode and ZeroNodeAttr
    """
    if not aug_spec:
        return Identity()
    
    # If the spec starts with "TP", it's a two-part augmentation
    if aug_spec.startswith("TP:"):
        parts = aug_spec.split(":", 1)
        if len(parts) == 2:
            # The rest of the spec after "TP:" is the original augmentation
            original_aug = get_two_part_aug(parts[1], node_feature_distribution, seed)
        else:
            # If no ":" is present, the rest of the spec is the original augmentation
            original_aug = get_two_part_aug(aug_spec[2:], node_feature_distribution, seed)
        
        # Return a TwoPartAugmentation with the original augmentation and default percentage
        return TwoPartAugmentation(original_aug, 0.5, seed)
    
    # Process global and local augmentations
    global_augs = []
    local_augs = []
    
    for spec in aug_spec.split(","):
        if spec.startswith("G"):
            # Global augmentation
            spec = spec[1:]
            if spec.startswith("ND"):
                global_augs.append(GlobalDropNode(float(spec[2:]), seed))
            elif spec.startswith("NZ"):
                global_augs.append(GlobalZeroNodeAttr(float(spec[2:]), seed))
            elif spec.startswith("NR"):
                if node_feature_distribution is None:
                    raise ValueError(f"node_feature_distribution not defined for GlobalRandomNodeAttr")
                global_augs.append(GlobalRandomNodeAttr(node_feature_distribution, float(spec[2:]), seed))
            else:
                raise ValueError(f"Unknown global augmentation {spec}")
        else:
            # Use the original augmentations from get_aug for local specs
            if spec.startswith("ND"):
                local_augs.append(DropNode(float(spec[2:])))
            elif spec.startswith("NZ"):
                local_augs.append(ZeroNodeAttr(float(spec[2:])))
            elif spec.startswith("NR"):
                if node_feature_distribution is None:
                    raise ValueError(f"node_feature_distribution not defined for RandomNodeAttr")
                local_augs.append(RandomNodeAttr(node_feature_distribution, float(spec[2:])))
            else:
                raise ValueError(f"Unknown augmentation {spec}")
    
    augs = global_augs + local_augs
    
    if len(augs) == 1:
        return augs[0]
    else:
        return Compose(augs)