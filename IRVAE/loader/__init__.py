from torch.utils import data
import numpy as np
import torch
from loader.MNIST_dataset import MNIST

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict)
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True)
    )
    return loader

def get_dataset(data_dict):
    name = data_dict["dataset"]
    if name == 'MNIST':
        dataset = MNIST(**data_dict)
    else:
        print(f"Loading {name} dataset...")
        cond_joint = np.load(f'./dataset/{name}/manifold/data_fixed_50000.npy')
        nulls     = np.load(f'./dataset/{name}/manifold/null_fixed_50000.npy')
        label     = np.load(f'./dataset/{name}/manifold/label_fixed_50000.npy')
        cond      = cond_joint[:, :3].astype(np.float32)
        joint     = cond_joint[:,3:].astype(np.float32)
        nulls     = nulls.astype(np.float32)
        label     = label.astype(np.float32)
        dataset   = data.TensorDataset(
            torch.from_numpy(joint),
            # torch.from_numpy(nulls),
            torch.from_numpy(label),
            # torch.from_numpy(cond)
        )
    return dataset