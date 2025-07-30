import torch
import torch.nn as nn
import math


class InstanceLoss(nn.Module):
    # latent space distance loss
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    # Cluster loss task space pose loss
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

class SupervisedPoseLoss(nn.Module):
    def __init__(self, class_num=None, device=None, loss_type="geodesic"):
        super(SupervisedPoseLoss, self).__init__()
        self.loss_type = loss_type
        self.device = device
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, c_i, c_j, pose):
        pose = pose.float().to(c_i.device) 

        if self.loss_type == "regression":
            loss_i = self.criterion(c_i, pose)
            loss_j = self.criterion(c_j, pose)

            # Total loss is the average of the two losses from both views
            total_loss = (loss_i + loss_j) / 2
        
        elif self.loss_type == "geodesic":
            def geodesic_distance(pose1, pose2):
                # Calculate the geodesic distance between two SE(3) matrices
                rotation_diff = torch.matmul(pose1[:, :3, :3], pose2[:, :3, :3].transpose(1, 2))
                trace = torch.diagonal(rotation_diff, dim1=-2, dim2=-1).sum(-1)
                rotation_distance = torch.arccos((trace - 1) / 2)
                
                translation_distance = torch.norm(pose1[:, :3, 3] - pose2[:, :3, 3], dim=-1)
                
                return rotation_distance + translation_distance

            geodesic_loss_i = geodesic_distance(c_i, pose)
            geodesic_loss_j = geodesic_distance(c_j, pose)

            # Total loss is the average of the two losses from both views
            total_loss = (geodesic_loss_i + geodesic_loss_j) / 2

        elif self.loss_type == "se3":
            # Assuming pose is a 4x4 transformation matrix and c_i, c_j are predicted 4x4 transformation matrices
            def se3_distance(pose1, pose2):
                # Calculate the Frobenius norm of the difference between the two transformation matrices
                return torch.norm(pose1 - pose2, p='fro')

            se3_loss_i = se3_distance(c_i, pose)
            se3_loss_j = se3_distance(c_j, pose)

            # Total loss is the average of the two losses from both views
            total_loss = (se3_loss_i + se3_loss_j) / 2

        return total_loss

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


class HMLC(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmce', batch_size=0):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(temperature)
        self.loss_type = loss_type
        self.batch_size = batch_size

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        """
        Compute loss for model,
        Args:
            features: hidden vector of shape [2 * bsz, feature_dim].
            labels: ground truth of shape [bsz, [l1, l2, l3, ...]].
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        mask = torch.ones(labels.shape).to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))

        for l in range(1,labels.shape[1]):
            mask[:, labels.shape[1]-l:] = 0
            layer_labels = labels * mask
            layer_outlier = (layer_labels == -1).any(dim=1)
            outlier_mask = torch.ones((self.batch_size, self.batch_size)).to(device)
            for i in range(self.batch_size):
                if layer_outlier[i]:
                    outlier_mask[i, :] = 0
                    outlier_mask[:, i] = 0
                    outlier_mask[i, i] = 1

            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)
            mask_labels = mask_labels * outlier_mask
            mask_labels = mask_labels.repeat(2, 2)
            features = features.view(features.shape[0], 1, -1)
            
            layer_loss = self.sup_con_loss(features, mask=mask_labels)

            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(
                  1/(l)).type(torch.float)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1/l).type(torch.float)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss')
            _, unique_indices = unique(layer_labels, dim=0)
            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss)
            # labels = labels[unique_indices]
            # mask = mask[unique_indices]
            # features = features[unique_indices]
        return cumulative_loss / labels.shape[1]

class HMLCSingleBatch(nn.Module):
    """
    Hierarchical Multi-Label Contrastive Loss for a single batch (no cross-batch memory).
    Designed for transformer-based models where all views are in a single batch.
    Assumes features shape: [2*bsz, feature_dim] (e.g., two views per sample).
    Labels shape: [bsz, num_levels] (multi-level cluster labels per sample).
    """
    def __init__(self, temperature=0.07, base_temperature=0.07, layer_penalty=None, loss_type='hmc', batch_size=256):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.layer_penalty = layer_penalty if layer_penalty is not None else (lambda x: 1.0)
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.sup_con_loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)

    def forward(self, features, labels):
        """
        Args:
            features: [2*bsz, feature_dim] (two views per sample, concatenated)
            labels: [bsz, num_levels] (multi-level cluster labels per sample)
        Returns:
            Scalar loss
        """
        device = features.device
        bsz = labels.shape[0]
        num_levels = labels.shape[1]
        cumulative_loss = torch.tensor(0.0, device=device)
        max_loss_lower_layer = torch.tensor(float('-inf'), device=device)

        # For each level in the hierarchy (from fine to coarse)
        for l in range(num_levels):
            # Mask out lower levels for current layer
            mask = torch.ones_like(labels, device=device)
            if l + 1 < num_levels:
                mask[:, l+1:] = 0
            layer_labels = labels * mask
            # Outlier detection: if any label at this level is -1, mark as outlier
            layer_outlier = (layer_labels == -1).any(dim=1)
            outlier_mask = torch.ones((bsz, bsz), device=device)
            for i in range(bsz):
                if layer_outlier[i]:
                    outlier_mask[i, :] = 0
                    outlier_mask[:, i] = 0
                    outlier_mask[i, i] = 1  # keep self

            # Build mask_labels: mask_labels[i, j] = 1 if all labels match (excluding outliers)
            mask_labels = torch.stack([
                torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                for i in range(bsz)
            ]).to(torch.uint8).to(device)
            mask_labels = mask_labels * outlier_mask
            # Repeat for both views (2*bsz)
            mask_labels = mask_labels.repeat(2, 2)

            # Reshape features for SupConLoss: [2*bsz, 1, feature_dim]
            features_ = features.view(features.shape[0], 1, -1)
            layer_loss = self.sup_con_loss(features_, mask=mask_labels)

            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(1/(l+1), dtype=torch.float, device=device)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer, layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer, layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(1/(l+1), dtype=torch.float, device=device)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss type: {}'.format(self.loss_type))
            max_loss_lower_layer = torch.max(max_loss_lower_layer, layer_loss)

        return cumulative_loss / num_levels


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss