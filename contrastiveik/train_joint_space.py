import os
import yaml
import numpy as np
import torch
import torchvision
import argparse

from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data

def train_jac():
    loss_epoch = 0
    for step, (x_i, x_null, c, jac, pose) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        jac = jac.to('cuda')
        pose = pose.to('cuda')
        
        batch_size = x_i.size(0)
        epsilon = torch.randn((batch_size, 7, 1)).to(x_i.device)
        identity = torch.eye(7).to(x_i.device)
        jac_T = jac.transpose(1, 2)
        jac_inv = torch.inverse(jac @ jac_T)

        x_j = x_i + (identity - jac_T @ jac_inv @ jac) @ epsilon
        x_j = x_j.to('cuda')

        z_i, z_j, c_i, c_j = model(x_i, x_j)

        # loss latent similarity
        loss_instance = criterion_instance(z_i, z_j)
        # loss pose similarity
        loss_pose = criterion_pose(c_i, c_j, pose)
        loss = loss_instance + loss_pose

        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t")
        loss_epoch += loss.item()
    return loss_epoch

def train():
    loss_epoch = 0
    for step, (x_i, x_null, c) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_null = x_null.to('cuda')
        c = c.to('cuda')
        
        batch_size = x_i.size(0)
        epsilon = torch.randn((batch_size, 8, 1)).to(x_i.device)

        x_j = x_i + torch.reshape(x_null @ epsilon, (batch_size, -1))
        x_j = x_j.to('cuda')

        z_i, z_j, c_i, c_j = model(x_i, x_j)

        # loss latent similarity
        loss = criterion_instance(z_i, z_j)
        # loss pose similarity
        # loss_pose = criterion_pose(c_i, c_j, pose)
        # loss = loss_instance + loss_pose

        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss.item()}\t")
        loss_epoch += loss.item()
    return loss_epoch

# arguments
parser = argparse.ArgumentParser()
config = yaml_config_hook("./contrastiveik/config/config_panda_dual_fixed.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))
args = parser.parse_args()

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# prepare data
if args.dataset == "panda_orientation":
    print("Loading panda orientation dataset")
    joint = np.load(f'./datasets/{args.exp_name}/manifold/data_{args.dataset_name}.npy')
    nulls = np.load(f'./datasets/{args.exp_name}/manifold/null_{args.dataset_name}.npy')
    jacobian = np.load(f'./datasets/{args.exp_name}/manifold/J0_{args.dataset_name}.npy')
    pose = np.load(f'./datasets/{args.exp_name}/manifold/Te_{args.dataset_name}.npy')

    max_data_len = len(joint)
    max_nulls_len = len(nulls)
    assert max_data_len == max_nulls_len

    C0 = np.array(joint[:,:args.c_dim],dtype=np.float32)[:max_data_len]
    D0 = np.array(joint[:,:],dtype=np.float32)[:max_data_len]
    N0 = np.array(nulls,dtype=np.float32)[:max_nulls_len]
    J0 = np.array(jacobian,dtype=np.float32)[:max_data_len]
    Te = np.array(pose,dtype=np.float32)[:max_data_len]

    dataset = data.TensorDataset(torch.from_numpy(D0), torch.from_numpy(N0), torch.from_numpy(C0), torch.from_numpy(J0), torch.from_numpy(Te))

elif args.dataset == "panda_dual_fixed":
    print("Loading panda dual fixed dataset")
    config = np.load(f'./dataset/{args.exp_name}/manifold/data_{args.dataset_name}.npy')
    nulls = np.load(f'./dataset/{args.exp_name}/manifold/null_{args.dataset_name}.npy')

    cond = np.array(config[:,:args.c_dim],dtype=np.float32)
    joint = np.array(config[:,args.c_dim:],dtype=np.float32)
    nulls = np.array(nulls,dtype=np.float32)

    max_data_len = len(joint)
    max_nulls_len = len(nulls)

    dataset = data.TensorDataset(torch.from_numpy(joint), torch.from_numpy(nulls), torch.from_numpy(cond))
    
    assert max_data_len == max_nulls_len

else:
    raise NotImplementedError

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.workers,
)

# initialize model
print("Loading model", args.resnet)
model = network.SimNetwork(input_dim=args.input_dim, feature_dim=args.feature_dim, 
                            instance_dim=args.instance_dim, cluster_dim=args.cluster_dim)
model = model.to('cuda')
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
if args.reload:
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
    checkpoint = torch.load(model_fp)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    args.start_epoch = checkpoint['epoch'] + 1
loss_device = torch.device("cuda")
criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(loss_device)
criterion_cluster = contrastive_loss.ClusterLoss(args.cluster_dim, args.cluster_temperature, loss_device).to(loss_device)
criterion_pose = contrastive_loss.SupervisedPoseLoss(loss_type="geodesic", device=loss_device).to(loss_device)

# train
for epoch in range(args.start_epoch, args.epochs):
    lr = optimizer.param_groups[0]["lr"]
    loss_epoch = train()
    if epoch % 100 == 0:
        save_model(args, model, optimizer, epoch)
    print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
save_model(args, model, optimizer, args.epochs)
