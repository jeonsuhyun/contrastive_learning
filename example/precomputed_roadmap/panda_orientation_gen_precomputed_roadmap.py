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

exp_name = 'panda_orientation'
data_len = 10000
max_num_edges = 20

model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=exp_name), 'r'), Loader=yaml.FullLoader)

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

constraint.set_max_iterations(500)
constraint.constraint.set_early_stopping(True)
pc = constraint.planning_scene

data = np.load(f'dataset/{exp_name}/manifold/data_50000.npy')

precomputed_roadmap = PrecomputedRoadmap(constraint)

precomputed_roadmap.compute(q_set=data[:data_len], max_num_edges=max_num_edges)

print(precomputed_roadmap.graph.number_of_nodes())
print(precomputed_roadmap.graph.number_of_edges())

precomputed_roadmap.save_graph(f'dataset/{exp_name}/manifold/precomputed_roadmap_{data_len}_{max_num_edges}.pkl')