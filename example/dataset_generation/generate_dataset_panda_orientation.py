from srmt.constraints.constraints import OrientationConstraint, DualArmConstraint, MultiChainConstraint
from srmt.utils.transform_utils import get_pose, get_transform
from srmt.planning_scene.planning_scene_tools import add_shelf, add_table
import numpy as np
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

from math import pi, cos, sin

import yaml
import argparse
import multiprocessing as mp
from scipy.linalg import null_space

from ljcmp.utils.model_utils import generate_constrained_config


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='panda_orientation', help='panda_orientation, panda_dual, panda_dual_orientation')
parser.add_argument('--dataset_size', '-D', type=int, default=10000)
parser.add_argument('--seed', type=int, default=1107)
parser.add_argument('--num_workers', '-J', type=int, default=8)
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

    R_offset = np.zeros((3,3))
    R_offset[2,0] = 1.0
    R_offset[1,1] = -1.0
    R_offset[0,2] = 1.0
    constraint = OrientationConstraint(arm_names=model_info['arm_names'],
                                        arm_dofs=model_info['arm_dofs'],
                                        base_link=model_info['base_link'],
                                        axis=0,
                                        orientation_offset=R_offset,
                                        ee_links=model_info['ee_links'],
                                        hand_names=model_info['hand_names'],
                                        hand_joints=model_info['hand_joints'],
                                        hand_open=model_info['hand_open'],
                                        hand_closed=model_info['hand_closed'])
    constraint.set_max_iterations(args.max_iter)
    constraint.set_early_stopping(True)
    pc = constraint.planning_scene
            
    # shelf_1_pos = np.array([0.8,0.0,0.75])
    # add_shelf(pc, shelf_1_pos, 0, -1.572, 0.7, 0.5, 1.5, 0.02, 4, 'shelf_1')                  
    
    start_pos_base = np.array([0.65, 0.0, 0.88]) 
    while True:
        q = np.random.uniform(constraint.lb, constraint.ub)
        r = constraint.solve_ik(q, start_pos_base)
        if r is False:
            continue
        if (q < constraint.lb).any() or (q > constraint.ub).any():
            continue
        if constraint.planning_scene.is_valid(q) is False:
            continue
        break

    pc.display(q)
    pc.add_cylinder('start', 0.1, 0.03, start_pos_base, [0,0,0,1])
    pc.attach_object('start', 'panda_hand',[])


    def set_constraint_by_condition(condition):
        pass

    return constraint, set_constraint_by_condition 


generate_constrained_config(constraint_setup_fn=set_constraint, 
                            exp_name=args.exp_name, 
                            workers_seed_range=range(args.seed, args.seed+args.num_workers), 
                            dataset_size=args.dataset_size, samples_per_condition=args.samples_per_condition,
                            save_top_k=args.save_top_k, save_every=args.save_every, display=args.display,
                            timeout=args.timeout, fixed_condition=np.empty(0))