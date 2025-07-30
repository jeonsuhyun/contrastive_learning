import os
import argparse
import numpy as np
import torch
import torch.utils.data as data
import yaml
import wandb

from math import sin, cos, pi
from scipy.spatial.transform import Rotation as R
from modules import network, contrastive_loss

def train_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    for step, (x_i, _, label, c) in enumerate(data_loader):
        x_i    = x_i.to(device)
        label  = label.flip(dims=[1]).to(device)
        c      = c.to(device)
        optimizer.zero_grad()
        
        # Single batch, no null, no augmentation
        z_i = model(x_i)
        # Use only z_i for single batch loss
        loss = criterion_hmlc_single_batch(z_i, label)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log batch metrics to W&B every 50 steps
        if step % 50 == 0:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/step": epoch * len(data_loader) + step
            })

    avg_loss = running_loss / len(data_loader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, args):
    ckpt_path = os.path.join(args.model_path, f"checkpoint_{epoch}.tar")
    torch.save({
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, ckpt_path)
    # Log as W&B artifact
    artifact = wandb.Artifact(
        name=f"model-checkpoint-{epoch}",
        type="model",
        description=f"Checkpoint at epoch {epoch}"
    )
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="ur5_dual_fixed_single", required=True,
                        help="Name of the experiment. Used to load the corresponding config YAML file.")

    # 먼저 experiment 이름만 받아서 config 경로 설정
    args_partial, _ = parser.parse_known_args()
    config_path = f"./contrastiveik/config/config_{args_partial.config_yaml}.yaml"

    # YAML 파일 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # config 내용을 parser에 추가
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # 0) Parse arguments
    args = parse_args()
    os.makedirs(args.model_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1) Initialize W&B
    wandb.init(
        project="contrastive-ik",
        name=args.exp_name,
        config={
            "learning_rate": args.learning_rate,
            "feature dimension": args.feature_dim,
            "instance dimension": args.instance_dim,
            "cluster dimension": args.cluster_dim,
            "dataset_name": args.dataset_name,
        }
    )

    # 2) Prepare dataset & loader (same as before)
    if "panda" in args.dataset:
        print("Loading panda dataset...")
        cond_joint = np.load(f'./dataset/{args.exp_name}/manifold/data_fixed_50000.npy')
        nulls     = np.load(f'./dataset/{args.exp_name}/manifold/null_fixed_50000.npy')
        label     = np.load(f'./dataset/{args.exp_name}/manifold/label_fixed_50000.npy')
        cond      = cond_joint[:, :args.c_dim].astype(np.float32)
        joint     = cond_joint[:,:].astype(np.float32)
        nulls     = nulls.astype(np.float32)
        label     = label.astype(np.float32)
        dataset   = data.TensorDataset(
            torch.from_numpy(joint),
            torch.from_numpy(nulls),
            torch.from_numpy(label),
            torch.from_numpy(cond)
        )
    
    elif "ur5" in args.dataset:
        print("Loading UR5 dataset...")
        cond_joint = np.load(f'./dataset/{args.exp_name}/manifold/data_fixed_50000.npy')
        nulls     = np.load(f'./dataset/{args.exp_name}/manifold/null_fixed_50000.npy')
        label     = np.load(f'./dataset/{args.exp_name}/manifold/data_fixed_50000_dbscan_joint_labels.npy')

        # import pdb; pdb.set_trace()

        cond      = cond_joint[:, :args.c_dim].astype(np.float32)
        joint     = cond_joint[:, args.c_dim:].astype(np.float32)
        nulls     = nulls.astype(np.float32)
        label     = label.astype(np.float32)
        dataset   = data.TensorDataset(
            torch.from_numpy(joint),
            torch.from_numpy(nulls),
            torch.from_numpy(label),
            torch.from_numpy(cond)
        )

    elif "tocabi" in args.dataset:
        print("Loading Tocabi dataset...")
        cond_joint = np.load(f'./dataset/{args.exp_name}/manifold/data_fixed_50000.npy')
        nulls     = np.load(f'./dataset/{args.exp_name}/manifold/null_fixed_50000.npy')
        label     = np.load(f'./dataset/{args.exp_name}/manifold/label_fixed_50000.npy')
        cond      = cond_joint[:, :args.c_dim].astype(np.float32)
        joint     = cond_joint[:, args.c_dim:].astype(np.float32)
        nulls     = nulls.astype(np.float32)
        label     = label.astype(np.float32)
        dataset   = data.TensorDataset(
            torch.from_numpy(joint),
            torch.from_numpy(nulls),
            torch.from_numpy(label),
            torch.from_numpy(cond)
        )

    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    # 3) Model, optimizer, criteria
    model = network.TransformerNetwork(
        input_dim=args.input_dim,
        feature_dim=args.feature_dim,
        instance_dim=args.instance_dim,
        cluster_dim=args.cluster_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    start_epoch = args.start_epoch

    # optional reload
    if args.reload:
        ckpt = torch.load(os.path.join(args.model_path, f"checkpoint_{start_epoch}.tar"))
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1

    criterion_hmlc_single_batch = contrastive_loss.HMLCSingleBatch(
        temperature=0.07,
        base_temperature=0.07,
        layer_penalty=None,
        loss_type='hmc',
        batch_size=args.batch_size
    ).to(device)
    # 4) Training loop with W&B logging
    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_epoch( model, data_loader, optimizer, device, epoch)

        # Log epoch metrics
        wandb.log({"train/epoch": epoch, "train/avg_loss": avg_loss,})

        print(f"Epoch [{epoch}/{args.epochs}]  avg_loss: {avg_loss:.6f}")

        # checkpoint & artifact every 100 epochs
        if epoch % 100 == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, epoch, args)

    # 5) Finish the run
    wandb.finish()



