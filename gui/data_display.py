import os
import yaml
import numpy as np
import torch
import time
import pickle
from functools import partial

import argparse
import matplotlib.pyplot as plt
import umap
import hdbscan

from sklearn.decomposition import PCA
from PyQt5.QtWidgets import QApplication, QPushButton, QSlider, QVBoxLayout, QLabel, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt
from ljcmp.utils.generate_environment import generate_environment
from ljcmp.utils.model_utils import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-E', type=str, default='ur5_dual', 
                    help='panda_orientation, panda_dual, panda_dual_orientation, panda_triple, tocabi, tocabi_orientation, ur5_dual')

args = parser.parse_args()

constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(args.exp_name)

constraint_model, _ = load_model(args.exp_name, model_info, load_validity_model=False)
pc = constraint.planning_scene

z_dim = model_info['z_dim']
c_dim = model_info['c_dim']

c_lb = model_info['c_lb']
c_ub = model_info['c_ub']

c_lb = np.array(c_lb)  
c_ub = np.array(c_ub)

app = QApplication([])
window = QWidget()
layout = QVBoxLayout()

label = QLabel()
label.setText('Latent code')
layout.addWidget(label)

latent_slider_list = []
latent_label_value_list = []

condition_slider_list = []
condition_label_value_list = []

z = np.zeros(z_dim)
c = (c_ub + c_lb) / 2
clustered_data = np.load(f'dataset/{args.exp_name}/manifold/data_fixed_50000.npy')
null_data = np.load(f'dataset/{args.exp_name}/manifold/null_fixed_50000.npy')

joint_data = clustered_data[:, 3:]
print(np.shape(joint_data))

# UMAP on joint data
# reducer = umap.UMAP(n_components=2, random_state=42)
# joint_umap = reducer.fit_transform(joint_data)

# # Run HDBSCAN clustering on the UMAP embedding
# clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10)
# umap_labels = clusterer.fit_predict(joint_umap)

# # Plot UMAP embedding colored by cluster
# plt.figure(figsize=(8, 6))
# palette = plt.get_cmap('tab20')
# unique_labels = set(umap_labels)
# for label in unique_labels:
#     mask = (umap_labels == label)
#     color = palette(label % 20) if label != -1 else (0.5, 0.5, 0.5, 0.5)
#     plt.scatter(joint_umap[mask, 0], joint_umap[mask, 1], s=5, alpha=0.7, color=color, label=f"Cluster {label}" if label != -1 else "Noise")
# plt.title("UMAP + HDBSCAN Clusters")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()


# Display joint data
for joint,null in zip(joint_data, null_data):
    print("joint", joint)
    joint = torch.tensor(joint, dtype=torch.float32)
    null = torch.tensor(null, dtype=torch.float32)
    # Generate multiple epsilons for null space exploration
    num_epsilons = 50  # or any number of samples you want
    epsilons = torch.randn(null.shape[-1], num_epsilons) * 0.1
    pc.display(np.array(joint))
    time.sleep(0.2)
    for i in range(num_epsilons):
        print(i)
        joint_null = joint + (null @ epsilons[:, i]).squeeze(-1)
        if pc.is_valid(np.array(joint_null)):
            pc.display(np.array(joint_null))
            time.sleep(0.2)
        else:
            print("invalid")