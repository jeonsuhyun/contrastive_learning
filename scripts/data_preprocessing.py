import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# from srmt.kinematics.trac_ik import TRACIK
# from ljcmp.utils.generate_environment import generate_environment
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3d projection

data = np.load(f'dataset/ur5_dual/manifold/data_fixed_50000.npy')

# joint part만 추출 (앞 3개는 c, 나머지는 joint)
joint_data = data[:, 3:]

right_joint_data = joint_data[:, :6]
left_joint_data = joint_data[:, 6:]

print(f"Data shapes:")
print(f"Combined joint data: {joint_data.shape}")
print(f"Right joint data: {right_joint_data.shape}")
print(f"Left joint data: {left_joint_data.shape}")

# UMAP으로 2D 임베딩 - 세 가지 경우 모두 시도
from sklearn.manifold import TSNE

# UMAP reducers
umap_reducers = {
    "combined": umap.UMAP(n_components=2, random_state=42),
    "right": umap.UMAP(n_components=2, random_state=42),
    "left": umap.UMAP(n_components=2, random_state=42)
}

umap_embeddings = {
    "combined": umap_reducers["combined"].fit_transform(joint_data),
    "right": umap_reducers["right"].fit_transform(right_joint_data),
    "left": umap_reducers["left"].fit_transform(left_joint_data)
}

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))

# UMAP plots
axes2[0,0].scatter(umap_embeddings["combined"][:, 0], umap_embeddings["combined"][:, 1], s=2, alpha=0.7)
axes2[0,0].set_title("UMAP: Combined Joint Data")
axes2[0,0].set_xlabel("UMAP 1")
axes2[0,0].set_ylabel("UMAP 2")

axes2[0,1].scatter(umap_embeddings["right"][:, 0], umap_embeddings["right"][:, 1], s=2, alpha=0.7)
axes2[0,1].set_title("UMAP: Right Joint Data")
axes2[0,1].set_xlabel("UMAP 1")
axes2[0,1].set_ylabel("UMAP 2")

axes2[0,2].scatter(umap_embeddings["left"][:, 0], umap_embeddings["left"][:, 1], s=2, alpha=0.7)
axes2[0,2].set_title("UMAP: Left Joint Data")
axes2[0,2].set_xlabel("UMAP 1")
axes2[0,2].set_ylabel("UMAP 2")

# t-SNE reducers
tsne_reducers = {
    "combined": TSNE(n_components=2, random_state=42),
    "right": TSNE(n_components=2, random_state=42),
    "left": TSNE(n_components=2, random_state=42)
}

tsne_embeddings = {
    "combined": tsne_reducers["combined"].fit_transform(joint_data),
    "right": tsne_reducers["right"].fit_transform(right_joint_data),
    "left": tsne_reducers["left"].fit_transform(left_joint_data)
}

# t-SNE plots
axes2[1,0].scatter(tsne_embeddings["combined"][:, 0], tsne_embeddings["combined"][:, 1], s=2, alpha=0.7)
axes2[1,0].set_title("t-SNE: Combined Joint Data")
axes2[1,0].set_xlabel("t-SNE 1")
axes2[1,0].set_ylabel("t-SNE 2")

axes2[1,1].scatter(tsne_embeddings["right"][:, 0], tsne_embeddings["right"][:, 1], s=2, alpha=0.7)
axes2[1,1].set_title("t-SNE: Right Joint Data")
axes2[1,1].set_xlabel("t-SNE 1")
axes2[1,1].set_ylabel("t-SNE 2")

axes2[1,2].scatter(tsne_embeddings["left"][:, 0], tsne_embeddings["left"][:, 1], s=2, alpha=0.7)
axes2[1,2].set_title("t-SNE: Left Joint Data")
axes2[1,2].set_xlabel("t-SNE 1")
axes2[1,2].set_ylabel("t-SNE 2")

plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans

# 클러스터 개수는 4로 설정
n_clusters = 4

# 세 가지 경우에 대해 클러스터링 수행
kmeans_combined = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_right = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_left = KMeans(n_clusters=n_clusters, random_state=42)

cluster_labels_combined = kmeans_combined.fit_predict(combined_joint_umap)
cluster_labels_right = kmeans_right.fit_predict(right_joint_umap)
cluster_labels_left = kmeans_left.fit_predict(left_joint_umap)

# 시각화 - 세 가지 경우 비교
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Combined joint data clustering
scatter1 = axes[0,0].scatter(combined_joint_umap[:, 0], combined_joint_umap[:, 1], 
                             c=cluster_labels_combined, s=2, alpha=0.7, cmap='tab10')
axes[0,0].set_title("Combined Joint Data (Left + Right)")
axes[0,0].set_xlabel("UMAP 1")
axes[0,0].set_ylabel("UMAP 2")

# Right joint data clustering
scatter2 = axes[0,1].scatter(right_joint_umap[:, 0], right_joint_umap[:, 1], 
                             c=cluster_labels_right, s=2, alpha=0.7, cmap='tab10')
axes[0,1].set_title("Right Joint Data Only")
axes[0,1].set_xlabel("UMAP 1")
axes[0,1].set_ylabel("UMAP 2")

# Left joint data clustering
scatter3 = axes[0,2].scatter(left_joint_umap[:, 0], left_joint_umap[:, 1], 
                             c=cluster_labels_left, s=2, alpha=0.7, cmap='tab10')
axes[0,2].set_title("Left Joint Data Only")
axes[0,2].set_xlabel("UMAP 1")
axes[0,2].set_ylabel("UMAP 2")

# 클러스터 간 일치도 분석
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Combined vs Right
ari_combined_right = adjusted_rand_score(cluster_labels_combined, cluster_labels_right)
nmi_combined_right = normalized_mutual_info_score(cluster_labels_combined, cluster_labels_right)

# Combined vs Left
ari_combined_left = adjusted_rand_score(cluster_labels_combined, cluster_labels_left)
nmi_combined_left = normalized_mutual_info_score(cluster_labels_combined, cluster_labels_left)

# Right vs Left
ari_right_left = adjusted_rand_score(cluster_labels_right, cluster_labels_left)
nmi_right_left = normalized_mutual_info_score(cluster_labels_right, cluster_labels_left)

# 클러스터 일치도 시각화
metrics_data = [
    ['Combined vs Right', ari_combined_right, nmi_combined_right],
    ['Combined vs Left', ari_combined_left, nmi_combined_left],
    ['Right vs Left', ari_right_left, nmi_right_left]
]

metrics_df = pd.DataFrame(metrics_data, columns=['Comparison', 'ARI', 'NMI'])
print("\nCluster Agreement Metrics:")
print(metrics_df)

# 클러스터 일치도 히트맵
axes[1,0].text(0.5, 0.5, f'Combined vs Right\nARI: {ari_combined_right:.3f}\nNMI: {nmi_combined_right:.3f}', 
                ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
axes[1,0].set_title("Combined vs Right Agreement")
axes[1,0].axis('off')

axes[1,1].text(0.5, 0.5, f'Combined vs Left\nARI: {ari_combined_left:.3f}\nNMI: {nmi_combined_left:.3f}', 
                ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
axes[1,1].set_title("Combined vs Left Agreement")
axes[1,1].axis('off')

axes[1,2].text(0.5, 0.5, f'Right vs Left\nARI: {ari_right_left:.3f}\nNMI: {nmi_right_left:.3f}', 
                ha='center', va='center', transform=axes[1,2].transAxes, fontsize=12)
axes[1,2].set_title("Right vs Left Agreement")
axes[1,2].axis('off')

plt.tight_layout()
plt.show()

# 각 클러스터의 특성 분석
print("\nCluster Analysis:")
for cluster_id in range(n_clusters):
    print(f"\nCluster {cluster_id}:")
    
    # Combined clustering
    combined_idx = np.where(cluster_labels_combined == cluster_id)[0]
    print(f"  Combined clustering: {len(combined_idx)} samples")
    
    # Right clustering
    right_idx = np.where(cluster_labels_right == cluster_id)[0]
    print(f"  Right clustering: {len(right_idx)} samples")
    
    # Left clustering
    left_idx = np.where(cluster_labels_left == cluster_id)[0]
    print(f"  Left clustering: {len(left_idx)} samples")

# Joint space 분석 - 각 클러스터의 평균 joint configuration
print("\n=== Joint Space Analysis ===")

for method_name, cluster_labels in [("Combined", cluster_labels_combined), 
                                   ("Right", cluster_labels_right), 
                                   ("Left", cluster_labels_left)]:
    print(f"\n{method_name} Clustering - Average Joint Configurations:")
    
    for cluster_id in range(n_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]
        if len(idx) == 0:
            continue
            
        if method_name == "Combined":
            cluster_joints = joint_data[idx]
            right_mean = np.mean(cluster_joints[:, :6], axis=0)
            left_mean = np.mean(cluster_joints[:, 6:], axis=0)
            print(f"  Cluster {cluster_id}: Right mean = {right_mean}, Left mean = {left_mean}")
        elif method_name == "Right":
            cluster_joints = right_joint_data[idx]
            right_mean = np.mean(cluster_joints, axis=0)
            print(f"  Cluster {cluster_id}: Right mean = {right_mean}")
        else:  # Left
            cluster_joints = left_joint_data[idx]
            left_mean = np.mean(cluster_joints, axis=0)
            print(f"  Cluster {cluster_id}: Left mean = {left_mean}")

# 클러스터 간 overlap 분석
print("\n=== Cluster Overlap Analysis ===")

# Combined clustering을 기준으로 다른 클러스터링과의 overlap 분석
for other_method, other_labels in [("Right", cluster_labels_right), ("Left", cluster_labels_left)]:
    print(f"\nCombined vs {other_method} clustering overlap:")
    
    for combined_cluster in range(n_clusters):
        combined_idx = np.where(cluster_labels_combined == combined_cluster)[0]
        
        # 각 combined 클러스터에 속한 샘플들이 다른 클러스터링에서 어떻게 분포하는지
        other_cluster_distribution = {}
        for sample_idx in combined_idx:
            other_cluster_id = other_labels[sample_idx]
            other_cluster_distribution[other_cluster_id] = other_cluster_distribution.get(other_cluster_id, 0) + 1
        
        print(f"  Combined Cluster {combined_cluster} -> {other_method} distribution: {other_cluster_distribution}")

print("Analysis complete!")

