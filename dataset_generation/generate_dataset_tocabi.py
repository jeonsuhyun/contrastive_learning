from srmt.constraints.constraints import OrientationConstraint, MultiChainConstraint
from srmt.utils.transform_utils import get_pose, get_transform
import numpy as np
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

from math import pi, cos, sin

import yaml
import argparse

from ljcmp.utils.model_utils import generate_constrained_config


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='tocabi', help='panda_orientation, panda_dual, panda_dual_orientation')
parser.add_argument('--dataset_size', type=int, default=10000)
parser.add_argument('--seed', type=int, default=1107)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--samples_per_condition', type=int, default=10)
parser.add_argument('--max_iter', type=int, default=500)
parser.add_argument('--save_every', type=int, default=-1, help='save every n data. -1 for not saving')
parser.add_argument('--save_top_k', type=int, default=1)
parser.add_argument('--timeout', type=float, default=0.2)
parser.add_argument('--display', type=bool, default=False)
parser.add_argument('--condition', nargs=3, type=float, default=[0.3, 0.05, 0.9], help='Fixed condition as three float values: d1 d2 theta')

args = parser.parse_args()

model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=args.exp_name), 'r'), Loader=yaml.FullLoader)

def set_constraint():
    # constraint should be prepared here for generating dataset

    constraint = MultiChainConstraint(arm_names=model_info['arm_names'],
                                        arm_dofs=model_info['arm_dofs'],
                                        base_link=model_info['base_link'],
                                        ee_links=model_info['ee_links'],
                                        hand_names=model_info['hand_names'],
                                        hand_joints=model_info['hand_joints'],
                                        hand_open=model_info['hand_open'],
                                        hand_closed=model_info['hand_closed'],
                                        planning_scene_name=model_info['planning_scene_name'])
                                        
    constraint.set_max_iterations(args.max_iter)
    pc = constraint.planning_scene
        
    def set_constraint_by_condition(condition):
        c = condition
        d1, d2, theta = c
        l = d1 + 2*d2*cos(theta)
        ly = l * sin(theta)
        lz = l * cos(theta)
        frame_rot = np.array([[0,1,0],[1,0,0],[0,0,-1]])
        
        dt = pi - 2 * theta
        chain_pos = np.array([0.0, ly, lz])
        chain_pos = chain_pos @ frame_rot
        chain_rot = np.array([[1, 0, 0], [0, cos(dt), -sin(dt)], [0, sin(dt), cos(dt)]])
        chain_rot = frame_rot @ chain_rot @frame_rot
        
        chain_quat = R.from_matrix(chain_rot).as_quat()

        t1 = np.concatenate([chain_pos, chain_quat])
        constraint.set_chains([t1])
        pc.detach_object('tray', 'L_Wrist2_Link')

        constraint.set_early_stopping(True)
        constraint.set_tolerance(1e-4)
        
        l_obj_z = d2 + d1/2 * cos(theta)
        l_obj_y = d1/2 * sin(theta)
        ee_to_obj_pos = np.array([0.0, l_obj_y, l_obj_z])
        ee_to_obj_pos = ee_to_obj_pos @ frame_rot
        
        obj_dt = -(pi/2 + theta)
        ee_to_obj_rot = np.array([[1, 0, 0], [0, cos(obj_dt), -sin(obj_dt)], [0, sin(obj_dt), cos(obj_dt)]])
        ee_to_obj_rot = frame_rot @ ee_to_obj_rot
        
        ee_to_obj_quat = R.from_matrix(ee_to_obj_rot).as_quat()
        
        q = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0])
        pos, quat = constraint.forward_kinematics('L_arm', q[:8])
        T_0g = get_transform(pos, quat)
        T_go = get_transform(ee_to_obj_pos, ee_to_obj_quat)
        T_0o = np.dot(T_0g, T_go)
        obj_pos, obj_quat = get_pose(T_0o)
        
        pc.add_box('tray', [d1 * 3/4, d1, 0.01], obj_pos, obj_quat)
        pc.update_joints(q)
        pc.attach_object('tray', 'L_Wrist2_Link', [])
        constraint.set_grasp_to_object_pose(go_pos=ee_to_obj_pos, go_quat=ee_to_obj_quat)

    return constraint, set_constraint_by_condition 

if args.condition is not None:
    c = np.array(args.condition, dtype=np.float32)
else:
    c = None

generate_constrained_config(constraint_setup_fn=set_constraint, 
                            exp_name=args.exp_name, 
                            workers_seed_range=range(args.seed, args.seed+args.num_workers), 
                            dataset_size=args.dataset_size, samples_per_condition=args.samples_per_condition,
                            save_top_k=args.save_top_k, save_every=args.save_every, display=args.display,
                            fixed_condition=c,
                            timeout=args.timeout)