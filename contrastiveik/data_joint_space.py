import numpy as np
import yaml
import argparse
import roboticstoolbox as rtb

"""load dataset"""
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='panda_orientation', help='panda_orientation, panda_dual, panda_dual_orientation, panda_triple')
parser.add_argument('--data_file_name', '-D', type=str, default='data_50000.npy')
parser.add_argument('--epsilon', type=float, default=3.5)
parser.add_argument('--min_samples', type=int, default=10)
parser.add_argument('--max_size', type=int, default=10000)
parser.add_argument('--surfix', '-S', type=str, default='')

args = parser.parse_args()

# dataset_name = '50000'
# data = np.load(f'./datasets/{args.exp_name}/manifold/data_{dataset_name}.npy')
# nulls = np.load(f'./datasets/{args.exp_name}/manifold/null_{dataset_name}.npy')
# Tes = np.load(f'./datasets/{args.exp_name}/manifold/Te_{dataset_name}.npy')
# print(Tes.shape)

robot = rtb.models.Panda()
print("panda.qr",robot.qr)
q = np.zeros((7,1))

j_0 = robot.jacob0(q)
j_e = robot.jacobe(q)
print("Jacobian_0: ", j_0.shape,"Jacobian_e", j_e.shape)
epsilon = np.random.rand(7,1)

q_j = q + (np.eye(7) - j_0.T @ np.linalg.inv(j_0 @ j_0.T) @ j_0) @ epsilon
q_0 = np.array([0.0, 0.1, 0.2, 0.2, 0.3, 0.5, 0.6])
print(q.T)
print(".A",robot.fkine(q).A)
print(". ",robot.fkine(q))
q_0_deg = np.degrees(q_0)
print(robot.fkine(q_0_deg).A)
print(q_j.T)
print(robot.fkine(q_j).A)
# Save the end-effector pose (Te) of the data
save = False
if save:
    Te = []
    for i in range(len(data)):
        q = data[i,:]  # Assuming the first 'n' columns are joint angles
        print(q)
        q_deg = np.degrees(q)  # Convert joint angles from radians to degrees
        Te.append(robot.fkine(q).A)  # Forward kinematics to get the end-effector pose

    Te = np.array(Te)
    np.save(f'./datasets/{args.exp_name}/manifold/Te_{dataset_name}.npy', Te)

save_jacobians = False
if save_jacobians:
    J0 = []
    # Je = []
    for i in range(len(data)):
        q = data[i,:]
        q_deg = np.degrees(q)
        J0.append(robot.jacobe(q))
        print(i)
        # Je.append(robot.jacob0(q))

    J0 = np.array(J0)
    # Je = np.array(Je)
    np.save(f'./datasets/{args.exp_name}/manifold/J0_{dataset_name}.npy', J0)
# model_info = yaml.load(open('ljcmp/model/{exp_name}/model_info.yaml'.format(exp_name=args.exp_name), 'r'), Loader=yaml.FullLoader)

# x_dim = model_info['x_dim']
# z_dim = model_info['z_dim']
# c_dim = model_info['c_dim']
# h_dim = model_info['constraint_model']['h_dim']

# max_data_len = len(data)
# max_nulls_len = len(nulls)
# assert max_data_len == max_nulls_len

# print("Data loaded")
# C0 = np.array(data[:,:c_dim],dtype=np.float32)[:max_data_len]
# D0 = np.array(data[:,c_dim:],dtype=np.float32)[:max_data_len]
# N0 = np.array(nulls,dtype=np.float32)[:max_data_len]
