import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pickle
import sklearn.datasets
import hdbscan
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3d projection
from itertools import combinations
import yaml
import umap.plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='tocabi', help='tocabi, tocabi_orientation')
    parser.add_argument('--data_type', type=str, default='ik_results', help='projected')
    parser.add_argument('--data_name', type=str, default='10000', help='fixed_50000, fixed_100000')
    args = parser.parse_args()
    model_info = yaml.load(open(f'model/{args.exp_name}/model_info.yaml', 'r'), Loader=yaml.FullLoader)
    
    if args.data_type == 'projected':
        data = np.load(f'dataset/{args.exp_name}/manifold/data_{args.data_name}.npy')
        null_data = np.load(f'dataset/{args.exp_name}/manifold/null_{args.data_name}.npy')
        joint_data = data[:, model_info['c_dim']:]
        print(f"Joint data: {joint_data.shape}")
        import pdb; pdb.set_trace()
    
    elif args.data_type == 'ik_results':
        if args.exp_name == 'tocabi':
            ik_dir = f'dataset/ik_results_tocabi_xyz_constrained_20250829_185646.pkl'
        elif args.exp_name == 'tocabi_orientation':
            ik_dir = f'dataset/ik_results_tocabi_z_only_20250829_011220.pkl'
        else:
            raise ValueError(f"Invalid exp_name: {args.exp_name}")
        data = []
        joint_data = []
        null_data = []
        jacobian_data = []
        with open(ik_dir, 'rb') as f:
            data = pickle.load(f)
        for i in range(len(data['joints'])):
            for j in range(len(data['joints'][i])):
                joint_data.append(np.array(data['joints'][i][j]))
                null_data.append(np.array(data['nulls'][i][j]))
                jacobian_data.append(np.array(data['jacobians'][i][j]))
            
    else:
        raise ValueError(f"Invalid data_type: {args.data_type}")

    joint_data = np.array(joint_data)
    null_data = np.array(null_data)
    jacobian_data = np.array(jacobian_data)
    import pdb; pdb.set_trace()
    print(f"Joint data: {joint_data.shape}")

    # UMAP parameter sets to try
    umap_params = [
        {"n_neighbors": 3, "min_dist": 0.1},
        {"n_neighbors": 4, "min_dist": 0.1},
        {"n_neighbors": 5, "min_dist": 0.1},
        {"n_neighbors": 6, "min_dist": 0.1},
        {"n_neighbors": 7, "min_dist": 0.1},
        {"n_neighbors": 8, "min_dist": 0.1},
        {"n_neighbors": 9, "min_dist": 0.1},
        {"n_neighbors": 10, "min_dist": 0.1},
        {"n_neighbors": 15, "min_dist": 0.1},
        {"n_neighbors": 20, "min_dist": 0.1},
    ]

    multi_label = np.zeros((joint_data.shape[0], len(umap_params)))
    all_embeddings = []
    all_labels = []
    print("constrained dimension: ", model_info['z_dim'])
    for i, params in enumerate(umap_params):  
        # UMAP embedding
        umap_reducer = umap.UMAP(
            n_components=model_info['z_dim'],
            random_state=42,
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist']
        )
        embedding = umap_reducer.fit_transform(joint_data)
        all_embeddings.append(embedding)

        # HDBSCAN clustering on UMAP embedding
        hdbscan_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=10,
            alpha=0.5,
            metric='euclidean',
            cluster_selection_epsilon=0.5,
            cluster_selection_method='eom'
        )
        labels = hdbscan_clusterer.fit_predict(embedding)
        all_labels.append(labels)
        multi_label[:, i] = labels

        # Print number of clusters (excluding noise)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        print(f"[{i}] HDBSCAN number of clusters (excluding noise): {n_clusters}")

        # Print clusters in descending order of number of points
        cluster_counts = []
        for cluster_label in unique_labels:
            if cluster_label == -1:
                continue  # skip noise
            n_points = np.sum(labels == cluster_label)
            cluster_counts.append((cluster_label, n_points))
        # Sort by n_points descending
        cluster_counts.sort(key=lambda x: x[1], reverse=True)
        for cluster_label, n_points in cluster_counts:
            print(f"Cluster {cluster_label}: {n_points} components")

    # Plot all UMAP embeddings together in a grid, colored by HDBSCAN cluster labels
    n_plots = len(umap_params)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
    axes = axes.flatten()
    for i, (embedding, labels, params) in enumerate(zip(all_embeddings, all_labels, umap_params)):
        ax = axes[i]
        if embedding.shape[1] >= 2:
            palette = sns.color_palette('tab20', np.unique(labels).max() + 1)
            colors = [palette[label] if label >= 0 else (0.5, 0.5, 0.5) for label in labels]
            ax.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.7, c=colors)
            ax.set_title(f"n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}")
            ax.set_xlabel("UMAP 0")
            ax.set_ylabel("UMAP 1")
        else:
            ax.text(0.5, 0.5, "z_dim < 2", ha='center', va='center')
            ax.set_title(f"n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}")
            ax.set_xticks([])
            ax.set_yticks([])
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

    print(multi_label.shape)
    np.save(
        f'dataset/{args.exp_name}/manifold/pseudo_labels.npy',
        multi_label,
        allow_pickle=True
    )
    np.save(f'dataset/{args.exp_name}/manifold/joint_data.npy', joint_data, allow_pickle=True)
    np.save(f'dataset/{args.exp_name}/manifold/null_data.npy', null_data, allow_pickle=True)
    np.save(f'dataset/{args.exp_name}/manifold/jacobian_data.npy', jacobian_data, allow_pickle=True)
