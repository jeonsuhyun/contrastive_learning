#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import os
import sys
import argparse

# Add the parent directory to the path to import ljcmp modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_path_result(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_robot_joint_data(data_path):
    data = np.load(data_path)
    ur5_dual_data = data[:, 3:]
    print(f"UR5 dual data shape: {ur5_dual_data.shape}")
    return ur5_dual_data

def prepare_data_for_umap(path_data, path_label, joint_data):
    """Prepare data for UMAP visualization."""
    mapper = umap.UMAP(n_neighbors=10, random_state=42, min_dist=0.1, n_components=6).fit(joint_data)
    joint_data_umap = mapper.transform(joint_data)
    path_data_umap = mapper.transform(path_data)

    combined_labels = np.concatenate([path_label, np.zeros(len(joint_data_umap))])
    combined_data_umap = np.vstack([path_data_umap, joint_data_umap])
    
    return combined_data_umap, combined_labels

def main():
    """Main function to load data and create visualization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ur5_dual")
    args = parser.parse_args()
    # File path
    if args.dataset == "ur5_dual":
        path_result_path = "result/emd_test_0617/ur5_dual/constrained_rrt/min_q/test_result.pkl"
        data_path = "dataset/ur5_dual/manifold/data_fixed_50000.npy"
        manifold_dim = 6
    elif args.dataset == "ur5_dual_orientation":
        path_result_path = "result/emd_test_0617/ur5_dual/constrained_rrt/min_q_z/test_result.pkl"
        data_path = "dataset/ur5_dual/manifold/data_fixed_50000.npy"
        manifold_dim = 4

    elif args.dataset == "panda_orientation":
        path_result_path = "result/emd_test_1/panda_orientation/latent_rrt/min_z/test_result.pkl"
        data_path = "dataset/panda_orientation/manifold/data_fixed_50000.npy"
        manifold_dim = 5
    elif args.dataset == "panda_dual":
        path_result_path = "result/emd_test_1/panda_dual/latent_rrt/min_q/test_result.pkl"
        data_path = "dataset/panda_dual/manifold/data_fixed_50000.npy"
        manifold_dim = 8
    elif args.dataset == "panda_dual_orientation":
        path_result_path = "result/emd_test_1/panda_dual_orientation/latent_rrt/min_q/test_result.pkl"
        data_path = "dataset/panda_dual_orientation/manifold/data_fixed_50000.npy"
        manifold_dim = 6
    elif args.dataset == "panda_triple":
        path_result_path = "result/emd_test_1/panda_triple/latent_rrt/min_z/test_result.pkl"
        data_path = "dataset/panda_triple/manifold/data_fixed_50000.npy"
        manifold_dim = 9
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    print("Loading robot joint data...")
    joint_data = load_robot_joint_data(data_path)
    manifold_dim = 2
    mapper = umap.UMAP(n_neighbors=10, random_state=42, min_dist=0.1, n_components=manifold_dim).fit(joint_data)
    joint_data_umap = mapper.transform(joint_data)

    print("Loading path result...")
    path_result_data = load_path_result(path_result_path)
    path_data = path_result_data['test_paths']

    print(f"Path data shape: {len(path_data)}")
    for i, path in enumerate(path_data):
        if path is not None:
            data = []
            for joint in path:
                data.append(joint)
            data = np.array(data)
            path_data_umap = mapper.transform(data)
            start_point = path_data_umap[0]
            goal_point = path_data_umap[-1]

            plt.scatter(joint_data_umap[:, 0], joint_data_umap[:, 1], c='lightgray', alpha=0.1, s=20, label='Joint Data')
            plt.scatter(start_point[0], start_point[1], c='red', s=40, marker='*', label='Start')
            plt.scatter(goal_point[0], goal_point[1], c='green', s=40, marker='X', label='Goal')
            
            # Create color gradient from green (start) to red (goal)
            colors = plt.cm.RdYlGn(np.linspace(0, 1, len(path_data_umap)-2))
            plt.scatter(path_data_umap[1:-1, 0], path_data_umap[1:-1, 1], c=colors, alpha=0.8, s=20)
            plt.legend()
            plt.show()
            plt.clf()
        
if __name__ == "__main__":
    main() 