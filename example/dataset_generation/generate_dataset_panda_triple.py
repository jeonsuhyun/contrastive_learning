from srmt.constraints.constraints import OrientationConstraint, DualArmConstraint, MultiChainConstraint
from srmt.utils.transform_utils import get_pose, get_transform
import numpy as np
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

from math import pi, cos, sin

import yaml
import argparse

from ljcmp.utils.model_utils import generate_constrained_config
from ljcmp.utils.cont_reader import ContinuousGraspCandidates


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='panda_triple', help='panda_orientation, panda_dual, panda_dual_orientation')
parser.add_argument('--dataset_size', type=int, default=10000)
parser.add_argument('--seed', type=int, default=1107)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--samples_per_condition', type=int, default=10)
parser.add_argument('--max_iter', type=int, default=500)
parser.add_argument('--save_every', type=int, default=-1, help='save every n data. -1 for not saving')
parser.add_argument('--save_top_k', type=int, default=1)
parser.add_argument('--timeout', type=float, default=0.2)
parser.add_argument('--display', type=bool, default=False)

args = parser.parse_args()

model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=args.exp_name), 'r'), Loader=yaml.FullLoader)

def set_constraint():
    # constraint should be prepared here for generating dataset
    constraint = MultiChainConstraint(arm_names=model_info['arm_names'], 
                                      base_link=model_info['base_link'], 
                                      arm_dofs=model_info['arm_dofs'],
                                      ee_links=model_info['ee_links'],
                                      hand_names=model_info['hand_names'], 
                                      hand_joints=model_info['hand_joints'],
                                      hand_open=model_info['hand_open'],
                                      hand_closed=model_info['hand_closed'],
                                      planning_scene_name=model_info['planning_scene_name']) 
    cgc = ContinuousGraspCandidates(file_name='model/{exp_name}/{cont_grasp}'.format(exp_name=args.exp_name, cont_grasp=model_info['cont_grasp']))

    pc = constraint.planning_scene


    chair_pos = np.array([0.69, 0.44, 1.19])
    chair_quat = np.array([0.9238795, -0.3826834, 0, 0])
    pc.add_mesh('chair',model_info['mesh'],
                chair_pos, chair_quat) # X-180, Z-45 Euler

    q_init = np.array([-0.12904, 0.173413, -0.390121, -1.30219, 0.0913822, 1.36203, 1.03038, 
                    -1.53953, -1.64972, 2.00178, -2.66883, 0.633282, 3.66834, 0.562251, 
                    -0.790644, -1.40522, 1.81529, -2.61019, -0.242376, 2.49991, 1.26293])
    p_cg, q_cg = cgc.get_global_grasp(model_info['c_idx'][2], 0.5, chair_pos, chair_quat)
    r, q = constraint.solve_arm_ik('panda_left', q_init[0:7], p_cg, q_cg)
    q_init[0:7] = q
    pc.update_joints(q_init)
    pc.attach_object('chair', 'panda_left_hand', [])

    def set_constraint_by_condition(y):
        # y: top right left ...
        constraint.planning_scene.detach_object('chair', 'panda_left_hand')
        p1, q1 = cgc.get_relative_transform(model_info['c_idx'][2], y[2], model_info['c_idx'][1], y[1])
        p2, q2 = cgc.get_relative_transform(model_info['c_idx'][2], y[2], model_info['c_idx'][0], y[0])
        p_cg, q_cg = cgc.get_global_grasp(model_info['c_idx'][2], y[2], chair_pos, chair_quat)
        t1 = np.concatenate([p1, q1])
        t2 = np.concatenate([p2, q2])
        constraint.set_chains([t1, t2])

        while True:
            r, q = constraint.solve_arm_ik('panda_left', q_init[0:7], p_cg, q_cg)
            if r: break
            
        q_init[0:7] = q
        pos, quat = cgc.get_grasp(model_info['c_idx'][2], y[2])
        T_og = get_transform(pos, quat)
        T_go = np.linalg.inv(T_og)
        constraint.set_grasp_to_object_pose(T_go=T_go)
        pc.update_object_pose('chair', chair_pos, chair_quat) # X-180, Z-45 Euler
        pc.update_joints(q_init)
        pc.attach_object('chair', 'panda_left_hand', [])

    return constraint, set_constraint_by_condition 


generate_constrained_config(constraint_setup_fn=set_constraint, 
                            exp_name=args.exp_name, 
                            workers_seed_range=range(args.seed, args.seed+args.num_workers), 
                            dataset_size=args.dataset_size, samples_per_condition=args.samples_per_condition,
                            save_top_k=args.save_top_k, save_every=args.save_every, 
                            timeout=args.timeout)