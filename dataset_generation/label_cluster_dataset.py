import numpy as np

import pickle, os
import pandas as pd
from sklearn.cluster import DBSCAN
import tqdm
from scipy.linalg import null_space
import argparse
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import hdbscan
import umap
from kneed import KneeLocator
from sklearn.manifold import TSNE

def make_joint_labels(args, dir_name): 
    """
    DBSCAN clustering
    """
    dataset = np.load(os.path.join(dir_name, args.data_file_name))
    # import pdb; pdb.set_trace()
    joint = dataset[:,3:]
    joint_label = np.zeros((joint.shape[0],11))

    if args.exp_name == 'panda_orientation':
        epsilon = np.linspace(0.2, 0.4, 10)

    elif args.exp_name == 'panda_dual':
        epsilon = np.linspace(1.5, 2.5, 11)

    elif args.exp_name == 'panda_dual_orientation':
        epsilon = np.linspace(1.0, 2.0, 11)

    elif args.exp_name == 'panda_triple':
        epsilon = np.linspace(3, 11, 11)
    
    elif args.exp_name == 'ur5_dual':
        epsilon = np.linspace(4.5, 6, 10)

    elif args.exp_name == 'ur5_dual_orientation':
        epsilon = np.linspace(1, 10, 10)
        
    elif args.exp_name == 'tocabi':
        epsilon = np.linspace(1, 10, 10)

    elif args.exp_name == 'tocabi_orientation':
        epsilon = np.linspace(1, 10, 10)

    else:
        epsilon = np.linspace(1, 10, 10)
        raise ValueError("Invalid exp_name")
    
    for i,eps in enumerate(epsilon):
        print('epsilon:', eps)
        model = DBSCAN(min_samples=args.min_samples, eps=eps).fit(joint)
        joint_label[:, i] = model.labels_
        labels = model.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('number of clusters:', n_clusters)
    
        # label_nums 계산
        if n_clusters > 0:
            label_nums = [np.sum(labels == j) for j in range(n_clusters)]
        else:
            label_nums = []

        print('label_nums:', label_nums)
        print('outliers:', np.sum(labels == -1))

    if args.save_data:
        model_path = os.path.join('dataset', args.exp_name, 'manifold')
        os.makedirs(model_path, exist_ok=True)
        
        file_name = args.data_file_name.replace('data', 'label')
        with open(os.path.join(model_path, file_name), 'wb') as f:
            np.save(f, joint_label)
        print(file_name,'saved')

def check_joint_labels(args):
    joint_label = np.load(os.path.join('dataset', args.exp_name, 'manifold', 'joint_label_50000.npy'))
    
    import pdb; pdb.set_trace() 


def hdbscan_clustering(args, dir_name):
    print(f"[INFO] Loading data from {args.data_file_name}...")
    dataset = np.load(os.path.join(dir_name, args.data_file_name))
    joint = dataset[:, 3:]

    print("[INFO] Standardizing data...")
    scaler = StandardScaler()
    joint_std = scaler.fit_transform(joint)

    print(f"[INFO] Running HDBSCAN with min_samples={args.min_samples}, min_cluster_size={args.epsilon}...")
    clusterer = hdbscan.HDBSCAN(min_samples=args.min_samples, min_cluster_size=int(args.epsilon), allow_single_cluster=True)
    labels = clusterer.fit_predict(joint_std)

    print(f"[INFO] Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters (and {np.sum(labels == -1)} noise points)")

    if args.save_data:
        save_name = args.data_file_name.replace(".npy", f"_hdbscan_labels{args.surfix}.npy")
        save_path = os.path.join(dir_name, save_name)
        np.save(save_path, labels)
        print(f"[INFO] Labels saved to {save_path}")
    
    # Optional: visualize clustering result (first 2 dims)
    import matplotlib.pyplot as plt
    plt.scatter(joint_std[:, 0], joint_std[:, 1], c=labels, cmap='tab20', s=2)
    plt.title("HDBSCAN Clustering Result (2D projection)")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()

def run_hdbscan(args, dir_name):
    print(f"[INFO] Loading data from {args.data_file_name}...")
    dataset = np.load(os.path.join(dir_name, args.data_file_name))
    joint = dataset[:, 3:]  # assuming joint space starts from dim 3

    print("[INFO] Standardizing data...")
    scaler = StandardScaler()
    joint_std = scaler.fit_transform(joint)

    print(f"[INFO] Running HDBSCAN with min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method='eom',
        allow_single_cluster=True
    )
    labels = clusterer.fit_predict(joint_std)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print(f"[RESULT] Clusters found: {n_clusters}, Noise points: {n_noise}")

    # 평가
    if n_clusters > 1:
        score = silhouette_score(joint_std[labels != -1], labels[labels != -1])
        print(f"[RESULT] Silhouette score (excluding noise): {score:.4f}")
    else:
        print("[RESULT] Silhouette score not computed (only one cluster).")

    # 저장
    if args.save_data:
        save_name = args.data_file_name.replace(".npy", f"_hdbscan_labels{args.surfix}.npy")
        np.save(os.path.join(dir_name, save_name), labels)
        print(f"[INFO] Labels saved to {save_name}")

    # 시각화
    if args.visualize:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
        joint_embedded = reducer.fit_transform(joint_std)

        plt.figure(figsize=(6, 5))
        plt.scatter(joint_embedded[:, 0], joint_embedded[:, 1], c=labels, cmap='tab20', s=2)
        plt.title("UMAP + HDBSCAN Clustering")
        plt.grid(True)
        plt.show()

def kneedle_dbscan(args, dir_name):
    k_distances = []
    print(f"[INFO] Loading data from {args.data_file_name}...")
    dataset = np.load(os.path.join(dir_name, args.data_file_name))
    joint = dataset[:, 3:]
    
    # import pdb; pdb.set_trace()
    k = args.min_samples
    print(f"[INFO] Calculating {k}-th nearest neighbor distances...")
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(joint)
    distances, _ = neigh.kneighbors(joint)
    k_distances = np.sort(distances[:, k-1])


    kneedle_convex_up = KneeLocator(
        range(len(k_distances)),
        k_distances,
        curve="convex",
        direction="increasing"
    )

    kneedle_convex_down = KneeLocator(
        range(len(k_distances)),
        k_distances,
        curve="convex",
        direction="decreasing"
    )

    kneedle_concave_up = KneeLocator(
        range(len(k_distances)),
        k_distances,
        curve="concave",
        direction="increasing"
    )

    kneedle_concave_down = KneeLocator(
        range(len(k_distances)),
        k_distances,
        curve="concave",
        direction="decreasing"
    )

    print(f"[INFO] Knee points found: {kneedle_convex_up.knee}, {kneedle_convex_down.knee}, {kneedle_concave_up.knee}, {kneedle_concave_down.knee}")


    modified_k_distance = k_distances[kneedle_concave_up.knee:kneedle_convex_up.knee+1]
    eps_min = np.percentile(modified_k_distance, 25)
    eps_max = np.percentile(modified_k_distance, 75)
    epsilon_range = np.linspace(eps_min, eps_max, 10)

    joint_label = np.zeros((joint.shape[0], len(epsilon_range)))
    for i, eps in enumerate(epsilon_range):
        print(f"[INFO] Running DBSCAN with eps={eps}...")
        model = DBSCAN(min_samples=args.min_samples, eps=eps).fit(joint)
        labels = model.labels_
        joint_label[:, i] = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"[INFO] DBSCAN eps={eps}: clusters={n_clusters}, outliers={np.sum(labels == -1)}")
    if args.save_data:
        save_name = args.data_file_name.replace(".npy", f"_dbscan_single_joint_labels{args.surfix}.npy")
        np.save(os.path.join(dir_name, save_name), joint_label)
        print(f"[INFO] Joint labels saved to {save_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ur5_dual', help='panda_orientation, panda_dual, panda_dual_orientation')
    parser.add_argument('--data_file_name', '-D', type=str, default='data_fixed_50000.npy')
    parser.add_argument('--epsilon', type=float, default=2)
    parser.add_argument('--min_cluster_size', type=int, default=2, help='min_cluster_size for HDBSCAN')
    parser.add_argument('--min_samples', type=int, default=8, help = " 2*data_dim or ln(data_size)")
    parser.add_argument('--max_size', type=int, default=10000)
    parser.add_argument('--surfix', '-S', type=str, default='')
    parser.add_argument('--save_data', '-s', type=bool, default=True)
    parser.add_argument('--visualize', '-v', type=bool, default=True, help='whether to visualize the clustering result')
    parser.add_argument('--mode', '-M', type=str, default='kneedle_dbscan', help='plot, make_joint_labels, check_joint_labels')

    args = parser.parse_args()
    dir_name = os.path.join('dataset', args.exp_name, 'manifold')

    if args.mode == "plot":
        k = args.min_samples  # 일반적으로 5~20
        dataset = np.load(os.path.join(dir_name,args.data_file_name))
        joint = dataset[:,3:]
        # import pdb; pdb.set_trace()
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(joint)
        distances, _ = neigh.kneighbors(joint)
        k_distances = np.sort(distances[:, k-1])

        plt.plot(k_distances)
        # plt.yscale("log")  # 고차원일수록 로그 스케일이 유리
        plt.title("k-distance graph")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{k}-th NN distance")
        plt.grid(True)
        plt.show()
    
    elif args.mode == "make_joint_labels":
        make_joint_labels(args, dir_name)
    
    elif args.mode == "hdbscan_clustering":
        run_hdbscan(args, dir_name)

    elif args.mode == "check_joint_labels":
        check_joint_labels(args)
    
    elif args.mode == "kneedle_dbscan":
        kneedle_dbscan(args, dir_name)

    else:
        raise ValueError("Invalid mode. Choose 'plot', 'make_joint_labels', or 'check_joint_labels'.")


