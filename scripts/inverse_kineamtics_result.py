import sys
import time
import torch
import os

import numpy as np
from optik import Robot, SolverConfig
from tracikpy import TracIKSolver
from srmt.kinematics.trac_ik import TRACIK
from contrastiveik.modules import resnet, network, transform
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

urdf_path = '/home/suhyun/catkin_ws/src/ljcmp/robots/panda.urdf'
base_name = 'panda_link0'
ee_name = 'panda_grasptarget'

robot = Robot.from_urdf_file(urdf_path, base_name, ee_name)
config = SolverConfig()

optik_total_time = 0
tracik_total_time = 0
hybridik_time = 0

optik_result = []
tracik_result = []
hybrid_result = []

q_target = np.random.uniform(*robot.joint_limits())
target_ee_pose = np.array(robot.fk(q_target))
x0 = np.random.uniform(*robot.joint_limits())

tracik_solver = TracIKSolver(
    '/home/suhyun/catkin_ws/src/ljcmp/robots/panda.urdf',
    'panda_link0',
    'panda_grasptarget'
)

N= 10000  # Number of attempts
for i in range(N):
    t0 = time.time()
    sol = robot.ik(config, target_ee_pose, x0)
    tf = time.time()

    if sol is not None:
        q_opt, c = sol
        optik_result.append(q_opt)
        optik_total_time += tf - t0

for i in range(N):
    t0 = time.time()
    qout = tracik_solver.ik(target_ee_pose, qinit=x0)
    tf = time.time()

    if qout is not None:
        tracik_result.append(qout)
        tracik_total_time += tf - t0

for i in range(N):
    t0 = time.time()
    sol, c = robot.ik(config, target_ee_pose, x0)
    qout = tracik_solver.ik(target_ee_pose, qinit=sol)
    tf = time.time()

    if qout is not None:
        hybrid_result.append(qout)
        hybridik_time += tf - t0

# Check IK result accuracy
optik_errors = []
tracik_errors = []
hybrid_errors = []

for q in optik_result:
    ee_pose = np.array(robot.fk(q))
    pos_err = np.linalg.norm(ee_pose[:3, 3] - target_ee_pose[:3, 3])
    rot_err = np.linalg.norm(ee_pose[:3, :3] - target_ee_pose[:3, :3])
    optik_errors.append((pos_err, rot_err))

for q in tracik_result:
    ee_pose = np.array(robot.fk(q))
    pos_err = np.linalg.norm(ee_pose[:3, 3] - target_ee_pose[:3, 3])
    rot_err = np.linalg.norm(ee_pose[:3, :3] - target_ee_pose[:3, :3])
    tracik_errors.append((pos_err, rot_err))

for q in hybrid_result:
    ee_pose = np.array(robot.fk(q))
    pos_err = np.linalg.norm(ee_pose[:3, 3] - target_ee_pose[:3, 3])
    rot_err = np.linalg.norm(ee_pose[:3, :3] - target_ee_pose[:3, :3])
    hybrid_errors.append((pos_err, rot_err))

optik_errors = np.array(optik_errors)
tracik_errors = np.array(tracik_errors)
hybrid_errors = np.array(hybrid_errors)

print(f"OptiK mean position error: {optik_errors[:,0].mean():.6f}, mean rotation error: {optik_errors[:,1].mean():.6f}")
print(f"TRAC-IK mean position error: {tracik_errors[:,0].mean():.6f}, mean rotation error: {tracik_errors[:,1].mean():.6f}")
print(f"HybridIK mean position error: {hybrid_errors[:,0].mean():.6f}, mean rotation error: {hybrid_errors[:,1].mean():.6f}")

# Reduce optik_result to 2D using UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
optik_result_umap = reducer.fit_transform(optik_result)

plt.figure(figsize=(8, 6))
plt.scatter(optik_result_umap[:, 0], optik_result_umap[:, 1], s=2, alpha=0.5, color='tab:orange')
plt.title("OptiK Result projected with UMAP")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()


# optik_result = np.array(optik_result)
# print(f"Found {len(optik_result)} solutions out of {N} attempts.")

# tracik_result = np.array(tracik_result)
# print(f"Found {len(tracik_result)} solutions out of {N} attempts.")

# hybrid_result = np.array(hybrid_result)
# print(f"Found {len(hybrid_result)} solutions out of {N} attempts.")

# print(f"OptiK total time: {optik_total_time:.4f} seconds")
# print(f"TRAC-IK total time: {tracik_total_time:.4f} seconds")
# print(f"HybridIK total time: {hybridik_time:.4f} seconds")

# plot_ik = True
# if plot_ik:
#     num_joints = len(optik_result[0])  # Number of joints
#     fig, axes = plt.subplots(num_joints, num_joints, figsize=(15, 15))
#     fig.suptitle("IK Result: Joint_i vs Joint_j")


#     for i in range(num_joints):
#         for j in range(num_joints):
#             ax = axes[i, j]
#             if i < j:
#                 # Upper triangle: OptiK
#                 ax.scatter(hybrid_result[:, i], hybrid_result[:, j], s=1, alpha=0.5, label='OptiK', color='tab:orange')
#             elif i > j:
#                 # Lower triangle: TRAC-IK
#                 ax.scatter(tracik_result[:, i], tracik_result[:, j], s=1, alpha=0.5, label='TRAC-IK', color='tab:blue')
#             else:
#                 # Diagonal: both for reference
#                 ax.scatter(hybrid_result[:, i], hybrid_result[:, j], s=1, alpha=0.5, color='tab:orange')
#                 ax.scatter(tracik_result[:, i], tracik_result[:, j], s=1, alpha=0.5, color='tab:blue')
#             if i == num_joints - 1:
#                 ax.set_xlabel(f'joint_{j+1}')
#             if j == 0:
#                 ax.set_ylabel(f'joint_{i+1}')
#             if i != num_joints - 1:
#                 ax.set_xticklabels([])
#             if j != 0:
#                 ax.set_yticklabels([])
#             if i == 0 and j == 0:
#                 ax.legend(markerscale=5)



#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.show()

# plot_z = False
# if plot_z:
#     model_path = os.path.join('./contrastiveik/save/panda_orientation/', 'checkpoint_3400.tar')
#     # model_path = os.path.join('./contrastiveik/save/panda_triple_fixed/', 'checkpoint_10000.tar')
#     input_dim=7
#     feature_dim=32
#     instance_dim=5
#     cluster_dim=2

#     model = network.SimNetwork(input_dim=input_dim, feature_dim=feature_dim, 
#                                 instance_dim=instance_dim, cluster_dim=cluster_dim)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.load_state_dict(torch.load(model_path, map_location=device.type)['net'])
#     model.to(device)
#     model.eval()

#     with torch.no_grad():
#         feature_path_z = model.inference(torch.tensor(tracik_result, dtype=torch.float32).to(device))

#     print("feature_path_z shape:", feature_path_z.shape)
#     print("feature_path_z sample values:\n", feature_path_z[:5].cpu().numpy())

#     # Plot feature_path_z with i,j component
#     feature_path_z_np = feature_path_z.cpu().numpy()
#     num_features = feature_path_z_np.shape[1]
#     fig, axes = plt.subplots(num_features, num_features, figsize=(15, 15))
#     fig.suptitle("feature_path_z: Feature_i vs Feature_j")

#     for i in range(num_features):
#         for j in range(num_features):
#             ax = axes[i, j]
#             if i != j:
#                 ax.scatter(feature_path_z_np[:, i], feature_path_z_np[:, j], s=1, alpha=0.5)
#             else:
#                 ax.hist(feature_path_z_np[:, i], bins=50, alpha=0.7)
#             if i == num_features - 1:
#                 ax.set_xlabel(f'feature_{j+1}')
#             if j == 0:
#                 ax.set_ylabel(f'feature_{i+1}')
#             if i != num_features - 1:
#                 ax.set_xticklabels([])
#             if j != 0:
#                 ax.set_yticklabels([])

#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.show()