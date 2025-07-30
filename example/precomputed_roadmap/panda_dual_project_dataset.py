import yaml
import numpy as np
from srmt.constraints.constraints import OrientationConstraint, DualArmConstraint, MultiChainConstraint, MultiChainFixedOrientationConstraint
from srmt.planning_scene.planning_scene import PlanningScene
from ljcmp.utils.cont_reader import ContinuousGraspCandidates

import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R
from ljcmp.planning.precomputed_roadmap import PrecomputedRoadmap

from math import pi, cos, sin
from srmt.utils.transform_utils import get_transform, get_pose

import tqdm

import yaml

np.set_printoptions(precision=3, suppress=True, linewidth=200)

exp_name = 'panda_dual'

data = np.load(f'dataset/{exp_name}_fixed/manifold/data_fixed_100000.npy')

model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=exp_name), 'r'), Loader=yaml.FullLoader)

if exp_name == 'panda_dual':
    constraint = MultiChainConstraint(arm_names=model_info['arm_names'],
                                        arm_dofs=model_info['arm_dofs'],
                                        base_link=model_info['base_link'],
                                        ee_links=model_info['ee_links'],
                                        hand_names=model_info['hand_names'],
                                        hand_joints=model_info['hand_joints'],
                                        hand_open=model_info['hand_open'],
                                        hand_closed=model_info['hand_closed'])
    
elif exp_name == 'panda_dual_orientation':
    constraint = MultiChainFixedOrientationConstraint(arm_names=model_info['arm_names'],
                                        arm_dofs=model_info['arm_dofs'],
                                        axis=2,
                                        base_link=model_info['base_link'],
                                        ee_links=model_info['ee_links'],
                                        hand_names=model_info['hand_names'],
                                        hand_joints=model_info['hand_joints'],
                                        hand_open=model_info['hand_open'],
                                        hand_closed=model_info['hand_closed'])
else:
    raise NotImplementedError

pc = constraint.planning_scene
def set_constraint(c):
    d1, d2, theta = c
    l = d1 + 2*d2*cos(theta)
    ly = l * sin(theta)
    lz = l * cos(theta)
    
    dt = pi - 2 * theta
    chain_pos = np.array([0.0, ly, lz])
    chain_rot = np.array([[1, 0, 0], [0, cos(dt), -sin(dt)], [0, sin(dt), cos(dt)]])
    chain_quat = R.from_matrix(chain_rot).as_quat()

    t1 = np.concatenate([chain_pos, chain_quat])
    constraint.set_chains([t1])
    pc.detach_object('tray', 'panda_2_hand_tcp')

    constraint.set_early_stopping(True)
    
    l_obj_z = d2 + d1/2 * cos(theta)
    l_obj_y = d1/2 * sin(theta)
    ee_to_obj_pos = np.array([0.0, l_obj_y, l_obj_z])
    obj_dt = -(pi/2 + theta)
    ee_to_obj_rot = np.array([[1, 0, 0], [0, cos(obj_dt), -sin(obj_dt)], [0, sin(obj_dt), cos(obj_dt)]])
    ee_to_obj_quat = R.from_matrix(ee_to_obj_rot).as_quat()

    q = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4, 0, 0, 0, -pi/2, 0, pi/2, pi/4])
    pos, quat = constraint.forward_kinematics('panda_arm_2', q[:7])
    T_0g = get_transform(pos, quat)
    T_go = get_transform(ee_to_obj_pos, ee_to_obj_quat)
    T_0o = np.dot(T_0g, T_go)
    obj_pos, obj_quat = get_pose(T_0o)

    pc.add_box('tray', [d1 * 3/4, d1, 0.01], obj_pos, obj_quat)
    pc.update_joints(q)
    pc.attach_object('tray', 'panda_2_hand_tcp', [])
    constraint.set_grasp_to_object_pose(go_pos=ee_to_obj_pos, go_quat=ee_to_obj_quat)

    
c = np.array([0.3, 0.05, 0.9],dtype=np.float32)
# print('[{}] getting new constraint set'.format(seed))
set_constraint(c)

projected_data = []
projection_failed = 0
validity_failed = 0
tq = tqdm.tqdm(data)
for qc in tq:
    c = qc[:3].astype(np.float64)
    q = qc[3:].astype(np.float64)
    # q = set_constraint(c)
    # print('c',c, '\nq:',q)
    # print(q.shape)
    # print(constraint.function(q))
    # input()
    tq.set_description('Projection failed: {}, Validity failed: {}'.format(projection_failed, validity_failed))
    r = constraint.project(q)
    
    if r is False:
        # print('No valid solution')
        projection_failed += 1
        continue

    if pc.is_valid(q) is False:
        # print('Invalid solution')
        validity_failed += 1
        continue

    projected_data.append(q)

projected_data = np.array(projected_data)
print('Projection failed: {}, Validity failed: {}'.format(projection_failed, validity_failed))

np.save('dataset/{exp_name}/manifold/projected_data_100000.npy'.format(exp_name=exp_name), projected_data)