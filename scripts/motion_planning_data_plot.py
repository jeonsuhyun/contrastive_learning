import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
import torch

from contrastiveik.modules import resnet, network, transform

# Example data

# Load data from /result/emd_test file
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-E', type=str, default='panda_triple', 
                    help='panda_orientation,'
                    'panda_dual,'
                    'panda_dual_orientation,'
                    'panda_triple')

args = parser.parse_args()
dir_path = os.path.join('./result/emd_test_1',args.exp_name,'latent_rrt')

path_q = []
path_r = []
scenario_names = ['min_z', 'min_q', 'random']

with open(f'{dir_path}/min_z/test_result.pkl', 'rb') as f:
    data = pickle.load(f)
    path_z = data['test_paths']

with open(f'{dir_path}/min_q/test_result.pkl', 'rb') as f:
    data = pickle.load(f)
    path_q = data['test_paths']

with open(f'{dir_path}/random/test_result.pkl', 'rb') as f:
    data = pickle.load(f)
    path_r = data['test_paths']


# # Load your neural network model (adjust the path and model class as needed)
model_path = os.path.join('./contrastiveik/save/panda_triple_fixed/', 'checkpoint_10000.tar')
input_dim=21 
feature_dim=128 
instance_dim=9
cluster_dim=12

model = network.SimNetwork(input_dim=input_dim, feature_dim=feature_dim, 
                               instance_dim=instance_dim, cluster_dim=cluster_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device.type)['net'])
model.to(device)
model.eval()

with torch.no_grad():
    feature_path_z = model.inference(torch.tensor(path_z[0], dtype=torch.float32).to(device))

print(f'Feature path z shape: {feature_path_z.shape}')

# # Plot each component of the feature vector along the path
# plt.figure(figsize=(15, 8))
# for i in range(feature_path_z.shape[1]):
#     plt.plot(feature_path_z[:, i].cpu().numpy(), label=f'Component {i+1}')
# plt.xlabel('Path Order')
# plt.ylabel('Feature Value')
# plt.title('Feature Components Along Path (min_z)')
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), ncol=1, fontsize='small')
# plt.tight_layout()
# plt.show()
fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# Plot features (from path_q) in the first subplot

# if path_q[j] is None or path_z[j] is None:
#     continue
for i in range(path_q[j].shape[1]):
    axs[0].plot(path_q[j][:, i], label=f'Feature {i+1}')
axs[0].set_ylabel('Feature Value')
axs[0].set_title('Feature Components Along Path (min_q)')
axs[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), ncol=2, fontsize='small')

# Plot originals (from path_z) in the second subplot
for i in range(path_z[j].shape[1]):
    axs[1].plot(path_z[j][:, i], '--', label=f'Original {i+1}')
axs[1].set_xlabel('Path Order')
axs[1].set_ylabel('Original Value')
axs[1].set_title('Original Components Along Path (min_z)')
axs[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), ncol=2, fontsize='small')

plt.tight_layout()
plt.show()
plt.close(fig)  # Clear the current figure for the next iteratio


# Plot each component of the original vector along the path
# plt.figure(figsize=(15, 8))
# for i in range(path_z[0].shape[1]):
#     plt.plot(path_z[0][:, i], label=f'Component {i+1}')
# plt.xlabel('Path Order')
# plt.ylabel('Original Value')
# plt.title('Original Vector Components Along Path (min_z)')
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), ncol=1, fontsize='small')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(15, 8))
# for i in range(path_q[0].shape[1]):
#     plt.plot(path_q[0][:, i], label=f'Component {i+1}')
# plt.xlabel('Path Order')
# plt.ylabel('Original Value')
# plt.title('Original Vector Components Along Path (min_q)')
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), ncol=1, fontsize='small')
# plt.tight_layout()
# plt.show()