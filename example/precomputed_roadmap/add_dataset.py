from srmt.constraints.constraints import OrientationConstraint, DualArmConstraint, MultiChainConstraint
from srmt.planning_scene.planning_scene import PlanningScene
from ljcmp.utils.cont_reader import ContinuousGraspCandidates

import numpy as np
import copy
import time
import scipy
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R
import os

from math import pi
import tqdm

import yaml

np.set_printoptions(precision=3, suppress=True, linewidth=200)

exp_name = 'panda_triple'

data = np.load(f'dataset/{exp_name}/manifold/projected_data_50000.npy')


exp_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=exp_name), 'r'), Loader=yaml.FullLoader)
cgc = ContinuousGraspCandidates(file_name='model/{exp_name}/{grasp}'.format(exp_name=exp_name, grasp=exp_info['cont_grasp']))


constraint = MultiChainConstraint(['panda_left', 'panda_right', 'panda_top'], 
                                'base', 
                                ['panda_left_hand_tcp', 
                                'panda_right_hand_tcp', 
                                'panda_top_hand_tcp'],
                                arm_dofs=exp_info['arm_dofs'],
                                hand_names=['hand_left', 'hand_right', 'hand_top'], 
                                hand_joints=[2, 2, 2], 
                                hand_open = [[0.0325,0.0325],[0.0325,0.0325] ,[0.0325,0.0325]], 
                                hand_closed = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
constraint.set_max_iterations(500)
constraint.constraint.set_early_stopping(True)
pc = constraint.planning_scene

chair_pos = np.array([0.69, 0.44, 1.19])
chair_quat = np.array([0.9238795, -0.3826834, 0, 0])
pc.add_mesh('chair','package://assembly_knowledge/models/meshes/ikea_stefan_without_bottom.stl',
            chair_pos, chair_quat) # X-180, Z-45 Euler

def set_constraint(c):
    # y: top right left ...
    # constraint.planning_scene.detach_object('chair', 'panda_left_hand')
    p1, q1 = cgc.get_relative_transform(exp_info['c_idx'][2], c[2], exp_info['c_idx'][1], c[1])
    p2, q2 = cgc.get_relative_transform(exp_info['c_idx'][2], c[2], exp_info['c_idx'][0], c[0])
    # p_cg, q_cg = cgc.get_global_grasp(exp_info['c_idx'][2], c[2], chair_pos, chair_quat)
    t1 = np.concatenate([p1, q1])
    t2 = np.concatenate([p2, q2])
    constraint.set_chains([t1, t2])
    first_attach = True

    while True:
        q = constraint.sample_valid(pc.is_valid, timeout=10.0)
        if q is False:
            # print('No valid solution')
            if not first_attach:
                pc.detach_object('chair', 'panda_left_hand')
            return False
        x = constraint.forward_kinematics('panda_left', q[0:7])
        T_x = cgc.get_transform(x[0], x[1])
        og = cgc.get_grasp(exp_info['c_idx'][2], c[2])
        T_og = cgc.get_transform(og[0], og[1])
        T_xo = np.matmul(T_x, np.linalg.inv(T_og))
        p_xo = T_xo[0:3,3]
        q_xo = st.Rotation.from_matrix(T_xo[0:3,0:3]).as_quat()

        pc.update_joints(q)
        if first_attach:
            pc.update_object_pose('chair', p_xo, q_xo) # X-180, Z-45 Euler
            pc.attach_object('chair', 'panda_left_hand', [])
            first_attach = False
        # pc.display(q)
        if pc.is_valid(q) is False:
            continue
        return q
    

c = np.array([0.3, 0.6, 0.3])
# print('[{}] getting new constraint set'.format(seed))
q = set_constraint(c)

for i in tqdm.tqdm(range(50000-len(data) + 1)):
    constraint.set_early_stopping(True)
    q = constraint.sample_valid(pc.is_valid, timeout=1e10)
    if q is False:
        print('No valid solution')
        continue
    data = np.vstack([data, q])

print('final:',len(data))
        
np.save('dataset/{exp_name}/manifold/projected_data_50000.npy'.format(exp_name=exp_name), data)