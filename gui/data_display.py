import os
import yaml
import numpy as np
import torch
import time
import pickle
from functools import partial

import argparse

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
print(np.shape(clustered_data))

# augmented_data = np.load(f'dataset/{args.exp_name}/manifold/data_fixed_50000_augmented.npy')

# for joint in augmented_data:
#     joint = torch.tensor(joint, dtype=torch.float32)
#     print("joint", joint)
#     pc.display(np.array(joint))
#     time.sleep(1)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Extract joint part from clustered_data (assuming first 3 are c, rest are joint)
joint_data = clustered_data[:, 3:]

# Run t-SNE on joint data
tsne = TSNE(n_components=2, random_state=42)
joint_tsne = tsne.fit_transform(joint_data)

plt.figure(figsize=(8, 6))
plt.scatter(joint_tsne[:, 0], joint_tsne[:, 1], s=2, alpha=0.5, color='tab:blue')
plt.title("t-SNE of Clustered Data (Joint Part)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()

import umap
from sklearn.decomposition import PCA

# PCA on joint data
pca = PCA(n_components=2)
joint_pca = pca.fit_transform(joint_data)

plt.figure(figsize=(8, 6))
plt.scatter(joint_pca[:, 0], joint_pca[:, 1], s=2, alpha=0.5, color='tab:green')
plt.title("PCA of Clustered Data (Joint Part)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

# UMAP on joint data
reducer = umap.UMAP(n_components=2, random_state=42)
joint_umap = reducer.fit_transform(joint_data)

plt.figure(figsize=(8, 6))
plt.scatter(joint_umap[:, 0], joint_umap[:, 1], s=2, alpha=0.5, color='tab:orange')
plt.title("UMAP of Clustered Data (Joint Part)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.show()


for cq,null in zip(clustered_data, null_data):
    c = cq[:3]
    joint = torch.tensor(cq[3:], dtype=torch.float32)
    null = torch.tensor(null, dtype=torch.float32)
    epsilon = torch.randn(null.shape[-1],1) * 0.01

    joint_null = joint + (null @ epsilon).squeeze(-1)

    print("joint", joint)
    set_constraint(c)
    pc.display(np.array(joint))
    time.sleep(0.1)
    print("joint_null", joint_null)
    pc.display(np.array(joint_null))
    time.sleep(0.1)