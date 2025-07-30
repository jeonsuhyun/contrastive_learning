import torch
import torch.nn as nn

if __name__ == "__main__":
    # Example usage
    # features = torch.randn(10, 128)  # Example feature vector
    # labels = torch.randint(0, 5, (10, 3))  # Example labels
    labels = torch.tensor([
        [-1,  0,  0],
        [-1,  2,  1],
        [ 1,  0, -1],
        [ 0,  0,  0],
        [-1,  0,  0],
        [-1,  0,  0],
        [-1,  2,  1],
        [ 1,  0, -1],
        [ 0,  0,  0],
        [-1,  0,  0],
    ])

    mask = torch.ones(labels.shape)
    cumulative_loss = torch.tensor(0.0)
    max_loss_lower_layer = torch.tensor(float('-inf'))
    
    for l in range(1,labels.shape[1]):
        mask[:, labels.shape[1]-l:] = 0
        print("mask",mask)

        layer_labels = labels * mask
        outlier_labels = (layer_labels == -1).any(dim=1)
        outlier_mask = torch.ones((layer_labels.shape[0], layer_labels.shape[0]))

        for i in range(layer_labels.shape[0]):
            if outlier_labels[i]:
                outlier_mask[i, :] = 0
                outlier_mask[:, i] = 0
                outlier_mask[i, i] = 1
        print("outlier_mask",outlier_mask)
        mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                    for i in range(layer_labels.shape[0])]).type(torch.uint8)
        mask_labels = mask_labels * outlier_mask
        print("mask_labels",mask_labels)