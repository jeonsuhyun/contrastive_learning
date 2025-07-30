import os
import argparse
import torch
import copy
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import skdim

from hnne import HNNE
from utils import yaml_config_hook
from modules import resnet, network, transform
from evaluation import evaluation
from torch.utils import data
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y, z) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config_joint_space.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "panda_orientation":
        print("Loading panda orientation dataset")
        joint = np.load(f'./datasets/{args.exp_name}/manifold/data_{args.dataset_name}.npy')
        nulls = np.load(f'./datasets/{args.exp_name}/manifold/null_{args.dataset_name}.npy')
        max_data_len = min(args.data_len, len(joint))
        max_nulls_len = min(args.data_len, len(nulls))

        assert max_data_len == max_nulls_len

        C0 = np.array(joint[:,:args.c_dim],dtype=np.float32)[:max_data_len]
        D0 = np.array(joint[:,:],dtype=np.float32)[:max_data_len]
        N0 = np.array(nulls,dtype=np.float32)[:max_nulls_len]
        dataset = data.TensorDataset(torch.from_numpy(D0), torch.from_numpy(N0), torch.from_numpy(C0))
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=200, #args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    model = network.SimNetwork(input_dim=args.input_dim, feature_dim=args.feature_dim, 
                               instance_dim=args.instance_dim, cluster_dim=args.cluster_dim)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    # Perform clustering: clustering_method = "kmeans", "dbscan", "contrastive_clustering"
    if args.clustering_method == "kmeans":
        print("### Performing K-means clustering ###")
        # Perform K-means clustering on D0
        num_clusters = args.num_clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(D0)
        cluster_labels = kmeans.labels_
        
        # Perform t-SNE on clustered data
        tsne = TSNE(n_components=2, random_state=0)
        D0_tsne = tsne.fit_transform(D0)

        # Plot t-SNE visualization of clustered data
        plt.scatter(D0_tsne[:, 0], D0_tsne[:, 1], c=cluster_labels)
        plt.title("t-SNE Visualization of kmeans Data")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()
    
    elif args.clustering_method == "dbscan":
        epsilon = args.epsilon
        while True:
            print('clustering with epsilon: ', epsilon)
            # Perform DBSCAN clustering on D0
            dbscan = DBSCAN(min_samples=args.min_samples, eps=epsilon).fit(D0)
            cluster_labels = dbscan.labels_
            print("Number of clusters: ", max(cluster_labels)+1)
            print("Number of outliers: ", len(cluster_labels[cluster_labels == -1]))
            label_nums = [len(cluster_labels[cluster_labels == i]) for i in range(max(cluster_labels)+1)]
            print('label_nums\n', label_nums)
            if max(label_nums) > args.max_size:            
                # Perform t-SNE on clustered data
                tsne = TSNE(n_components=2, random_state=0)
                D0_tsne = tsne.fit_transform(D0)

                # Plot t-SNE visualization of clustered data
                plt.scatter(D0_tsne[:, 0], D0_tsne[:, 1], c=cluster_labels)
                plt.title("t-SNE Visualization of dbscan Data")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.show()
                break

            print('size is too small, increasing epsilon')
            print('max size: ', max(label_nums))
            print('max label num: ', max(label_nums))
            epsilon += 0.1
        print(label_nums)
        print('max: ', np.argmax(label_nums))
        
    elif args.clustering_method == "contrastive_kmeans":
        print("### Feature extracor: contrastive learning ###")
        print("### Performing K-means clustering ###")
        X, Y = inference(data_loader, model, device)
        
        num_clusters = args.num_clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_

        # Perform t-SNE on feature vectors
        tsne = TSNE(n_components=2, random_state=0)
        X_tsne = tsne.fit_transform(D0) # X

        # Plot t-SNE visualization
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels)
        plt.title("t-SNE Visualization of Contrastive learning with kmeans")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    elif args.clustering_method == "contrastive_dbscan":
        print("### Feature extractor: contrastive learning ###")
        print("### Performing DBSCAN clustering ###")   
        X, Y = inference(data_loader, model, device)
        epsilon = args.epsilon_con
        
        print('clustering with epsilon: ', epsilon)
        # Perform DBSCAN clustering on X
        dbscan = DBSCAN(min_samples=args.min_samples, eps=epsilon).fit(X)
        cluster_labels = dbscan.labels_
        print("Number of clusters: ", max(cluster_labels)+1)
        print("Number of outliers: ", len(cluster_labels[cluster_labels == -1]))
        label_nums = [len(cluster_labels[cluster_labels == i]) for i in range(max(cluster_labels)+1)]
        print('label_nums\n', label_nums)
        if max(label_nums) > args.max_size:            
            # Perform t-SNE on clustered data
            tsne = TSNE(n_components=2, random_state=0)
            D0_tsne = tsne.fit_transform(D0)

            # Plot t-SNE visualization of clustered data
            plt.scatter(D0_tsne[:, 0], D0_tsne[:, 1], c=cluster_labels)
            plt.title("t-SNE Visualization of Contrastive learning with dbscan")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.show()
    elif args.clustering_method == "pca_kmeans":
        print("### Performing PCA and K-means clustering ###")
        pca = PCA(n_components=2)
        X = pca.fit_transform(D0)
        num_clusters = args.num_clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_

        tsne = TSNE(n_components=2, random_state=0)
        D0_tsne = tsne.fit_transform(D0)

        # Plot t-SNE visualization of clustered data
        plt.scatter(D0_tsne[:, 0], D0_tsne[:, 1], c=cluster_labels)
        plt.title("t-SNE Visualization of PCA with kmeans")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()
    elif args.clustering_method == "pca":
        X = torch.Tensor(D0)
        X = X - X.mean(dim=0, keepdim=True)
        eigh = torch.linalg.eigh(X.permute(1,0)@X)
        ratio = eigh.eigenvalues[-16:] / eigh.eigenvalues[-16:].sum()
        cusum_vals = torch.cumsum(ratio.sort(descending=True).values, dim=0)
        plt.figure(figsize=(8, 4))
        plt.scatter(torch.linspace(1,len(cusum_vals)+1,len(cusum_vals)), cusum_vals, c='tab:blue')
        plt.plot(torch.linspace(1,len(cusum_vals)+1,len(cusum_vals)), cusum_vals, c='tab:blue')
        plt.hlines(0.99, xmin=1,xmax=17, colors='tab:red',linestyles='--')
        plt.title("Global PCA")
        plt.show()

        print(skdim.id.TwoNN(0.1).fit_transform(X))
        print(skdim.id.FisherS().fit_transform(X))
        print(skdim.id.MADA().fit_transform(X))
        print(skdim.id.MLE().fit_transform(X))
    # elif args.clustering_method == "hnne":
    #     hnne = HNNE(dim=2)
    #     projection = hnne.fit_transform(D0)

    #     plt.figure(figsize=(8, 8))
    #     plt.scatter(*projection.T, s=1, targets)

    else:
        raise NotImplementedError

    # nmi, ari, f, acc = evaluation.evaluate(Y, X)
    # print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))