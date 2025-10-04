import numpy as np
import random
import torch

torch.multiprocessing.set_sharing_strategy("file_system") 

import sys
import os
torch.autograd.set_detect_anomaly(True)

sys.path.extend(os.path.join(os.path.dirname(__file__), "../../"))

from experiments.params import get_params
from experiments.trainer import TrainerFS

from data.data_loader_wrapper import get_dataset_wrap

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    torch.set_num_threads(4)

    params = get_params()
    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    if params["dataset"] in ["FB15K-237", "NELL", "ConceptNet", "Wiki"]:
        print("Using KG dataset - setting language model to sentence-transformers/all-mpnet-base-v2")
        params["bert_emb_model"] = "sentence-transformers/all-mpnet-base-v2"
    
    # Handle the centroids_path parameter for MAG240M dataset
    centroids_path = params.get("centroids_path", None)
    if centroids_path:
        # Normalize the path (remove any whitespace that might be introduced by command line continuation)
        centroids_path = centroids_path.strip()
        
        try:
            if os.path.exists(centroids_path):
                # Try to load the centroids to verify they're valid
                centroids = torch.load(centroids_path)
                print(f"Successfully loaded {len(centroids)} centroids from: {centroids_path}")
            else:
                print(f"Warning: Centroids path does not exist: {centroids_path}")
                print("Will use random sampling for all tasks")
                centroids_path = None
        except Exception as e:
            print(f"Error loading centroids from {centroids_path}: {e}")
            print("Will use random sampling for all tasks")
            centroids_path = None
    else:
        print("No centroids path provided. Will use random sampling for all tasks.")
    
    # Handle batch length caps
    if params["val_len_cap"] is None:
        print("Warning: val_len_cap is None, setting to default 100")
        params["val_len_cap"] = 100
    
    if params["test_len_cap"] is None:
        print("Warning: test_len_cap is None, setting to default 100")
        params["test_len_cap"] = 100
    
    datasets = get_dataset_wrap(
        root=params["root"],
        dataset=params["dataset"],
        force_cache=params["force_cache"],
        small_dataset=params["small_dataset"],
        invalidate_cache=None,
        original_features=params["original_features"],
        n_shot=params["n_shots"],
        n_query=params["n_query"],
        bert=None if params["original_features"] else params["bert_emb_model"],
        bert_device=params["device"],
        val_len_cap=params["val_len_cap"],
        test_len_cap=params["test_len_cap"],
        dataset_len_cap=params["dataset_len_cap"],
        n_way=params["n_way"],
        rel_sample_rand_seed=params["rel_sample_random_seed"],
        calc_ranks=params["calc_ranks"],
        kg_emb_model=params["kg_emb_model"] if params["kg_emb_model"] != "" else None,
        task_name=params["task_name"],
        shuffle_index=params["shuffle_index"],
        node_graph=params["task_name"] == "sn_neighbor_matching",
        use_subset=params.get("use_mag_subset", True) if params["dataset"] == "mag240m" else False,
        subset_size=params.get("mag_subset_size", 20000000) if params["dataset"] == "mag240m" else None,
        centroids_path=centroids_path
    )

    trnr = TrainerFS(datasets, params)

    trnr.train()