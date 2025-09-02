import os
import yaml
import numpy as np
import torch
import pickle
import argparse
import tqdm
import time
import matplotlib.pyplot as plt
from termcolor import colored
from ljcmp.utils.model_utils import benchmark, benchmark_ik_selection_all_pairs
from ljcmp.utils.generate_environment import generate_environment
from scipy.spatial.transform import Rotation as R
from srmt.kinematics.trac_ik import TRACIK
from srmt.constraints.constraints import MultiChainConstraint, MultiChainFixedOrientationConstraint
from srmt.planning_scene.planning_scene_tools import add_shelf, add_table

from contrastiveik.modules import network

from ljcmp.planning.constrained_bi_rrt import SampleBiasedConstrainedBiRRT
from ljcmp.planning.precomputed_roadmap import PrecomputedRoadmap, PrecomputedGraph
from ljcmp.planning.constrained_bi_rrt_latent_jump import ConstrainedLatentBiRRT
from ljcmp.planning.constrained_bi_rrt import ConstrainedBiRRT
from ljcmp.planning.sample_region import RegionSampler, LatentRegionSampler
from ljcmp.utils.time_parameterization import time_parameterize

def get_transform(pos, quat):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T

def get_pose_from_transform(T):
    pos = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat

def compute_grasp_pose(obj_pos, obj_quat, obj_to_ee_pos, obj_to_ee_quat):
    T_0o = get_transform(obj_pos, obj_quat)
    T_og = get_transform(obj_to_ee_pos, obj_to_ee_quat)
    T_0g = T_0o @ T_og
    return get_pose_from_transform(T_0g)

def compute_dual_grasp_poses(obj_pos, obj_quat, condition=None):
    if condition is None:
        condition = [0.3, 0.05, 0.9]
    d1, d2, theta = condition
    l_obj_z = d2 * np.sin(theta)
    l_obj_y = d1 / 2 + d2 * np.cos(theta)
    obj_to_ee_pos_l = np.array([0.0, l_obj_y, l_obj_z])
    obj_dt_l = (np.pi / 2 + theta)
    obj_to_ee_rot_l = np.array([
        [1, 0, 0],
        [0, np.cos(obj_dt_l), -np.sin(obj_dt_l)],
        [0, np.sin(obj_dt_l), np.cos(obj_dt_l)]
    ])
    obj_to_ee_quat_l = R.from_matrix(obj_to_ee_rot_l).as_quat()

    obj_to_ee_pos_r = np.array([0.0, -l_obj_y, l_obj_z])
    obj_dt_r = -(np.pi / 2 + theta)
    obj_to_ee_rot_r = np.array([
        [1, 0, 0],
        [0, np.cos(obj_dt_r), -np.sin(obj_dt_r)],
        [0, np.sin(obj_dt_r), np.cos(obj_dt_r)]
    ])
    obj_to_ee_quat_r = R.from_matrix(obj_to_ee_rot_r).as_quat()

    l_pos, l_quat = compute_grasp_pose(obj_pos, obj_quat, obj_to_ee_pos_l, obj_to_ee_quat_l)
    r_pos, r_quat = compute_grasp_pose(obj_pos, obj_quat, obj_to_ee_pos_r, obj_to_ee_quat_r)
    return (r_pos, r_quat), (l_pos, l_quat)

def compute_tocabi_grasp_poses(obj_pos, obj_quat, condition=None):
    """Compute grasp poses for dual arms given object pose and condition"""
    if condition is None:
        condition = [0.3, 0.05, 0.9]
    
    d1, d2, theta = condition
    
    # left arm offset
    l_obj_z = d2 * np.sin(theta)
    l_obj_y = d1/2 + d2 * np.cos(theta)

    # frame rotation
    frame_rot = R.from_euler('z', np.pi).as_matrix()
    obj_to_ee_pos_l = np.array([0.0, l_obj_y, l_obj_z])
    obj_to_ee_pos_l =  obj_to_ee_pos_l
    obj_dt_r = -(np.pi/2 - theta)
    obj_to_ee_rot_l = np.array([
        [1, 0, 0],
        [0, np.cos(obj_dt_r), -np.sin(obj_dt_r)],
        [0, np.sin(obj_dt_r),  np.cos(obj_dt_r)]
    ])
    obj_to_ee_rot_l =  obj_to_ee_rot_l @ frame_rot
    obj_to_ee_quat_l = R.from_matrix(obj_to_ee_rot_l).as_quat()

    # right arm offset
    obj_to_ee_pos_r = np.array([0.0, -l_obj_y, l_obj_z])
    obj_dt_l = (np.pi/2 - theta)
    obj_to_ee_rot_r = np.array([
        [1, 0, 0],
        [0, np.cos(obj_dt_l), -np.sin(obj_dt_l)],
        [0, np.sin(obj_dt_l),  np.cos(obj_dt_l)]
    ])
    obj_to_ee_rot_r = obj_to_ee_rot_r @ frame_rot
    obj_to_ee_quat_r = R.from_matrix(obj_to_ee_rot_r).as_quat()

    # Compute grasp poses
    T_0o = get_transform(obj_pos, obj_quat)      # Object in world
    T_og_l = get_transform(obj_to_ee_pos_l, obj_to_ee_quat_l)  # object to left ee
    T_og_r = get_transform(obj_to_ee_pos_r, obj_to_ee_quat_r)  # object to right ee         
    T_0g_r = T_0o @ T_og_r                            # right EE in world
    T_0g_l = T_0o @ T_og_l                            # left EE in world
    
    r_pos, r_quat = get_pose_from_transform(T_0g_r)
    l_pos, l_quat = get_pose_from_transform(T_0g_l)

    return (r_pos, r_quat), (l_pos, l_quat)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-E', type=str, default='ur5_dual', help='panda_orientation, panda_dual, panda_dual_orientation, panda_triple')
    parser.add_argument('--seed', type=int, default=1107)
    parser.add_argument('--use_given_start_goal', action='store_true', help='Use given start and goal')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--display', action='store_true', help='Enable display')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--method', '-M', type=str, default='constrained_rrt', help='latent_rrt, latent_rrt_latent_jump, sampling_rrt, precomputed_roadmap_prm, precomputed_graph_rrt, project_rrt')
    parser.add_argument('--test_scene_start_idx', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_test_scenes', type=int, default=100)
    parser.add_argument('--max_time', type=float, default=100.0)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--metric', type=str, default='min_q', help='Metric type: given, min_z, min_q, random')
    args = parser.parse_args()
    
    # Handle device selection
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print(colored('CUDA is not available, use CPU instead', 'red'))
    print(colored('Using device: {}'.format(args.device), 'green'))
    return args

def load_scene_and_set_start_goal(i, args, scene_dir, constraint):
    scene_dir_local = '{}/scene_{:04d}'.format(scene_dir, i)
    print(scene_dir_local)
    if not os.path.exists(f'{scene_dir_local}/scene.yaml'):
        print(f'{scene_dir_local}/scene.yaml not exist')
        return None, None, None, None, None, None

    scene = yaml.load(open(os.path.join(scene_dir_local, 'scene.yaml'), 'r'), Loader=yaml.FullLoader)
    t1_dim  = scene['table_1']['dim']
    t2_dim  = scene['table_2']['dim']
    add_table(constraint.planning_scene, 'table_1', scene['table_1']['pos'], t1_dim[0], t1_dim[1], t1_dim[2], t1_dim[3], t1_dim[4], t1_dim[5])
    add_table(constraint.planning_scene, 'table_2', scene['table_2']['pos'], t2_dim[0], t2_dim[1], t2_dim[2], t2_dim[3], t2_dim[4], t2_dim[5])
    condition = scene['c']
    d1, d2, theta = condition

    # load object start and goal pose
    obj_start_pose = np.array(scene['start_pose'])
    obj_goal_pose = np.array(scene['goal_pose'])

    # set feature map
    feature_map = args.metric != 'given'

    # set start and goal q
    if args.metric == 'min_z':
        if "ur5_dual" in args.exp_name:
            if "ur5_dual_orientation" in args.exp_name:
                model_path = os.path.join('contrastiveik','save',"ur5_dual_orientation_fixed", "checkpoint_10000.tar")
                input_dim=12
                feature_dim=64
                instance_dim=8
                cluster_dim=4
            else:
                model_path = os.path.join('contrastiveik','save',"ur5_dual_fixed_transformer", "checkpoint_10000.tar")
                input_dim=12
                feature_dim=64 
                instance_dim=6
                cluster_dim=6
        if "tocabi" in args.exp_name:
            if "tocabi_orientation" in args.exp_name:
                model_path = os.path.join('contrastiveik','save',"tocabi_orientation_fixed", "checkpoint_10000.tar")
                input_dim=16
                feature_dim=64
                instance_dim=8
                cluster_dim=8
            else:
                model_path = os.path.join('contrastiveik','save',"tocabi_fixed_transformer", "checkpoint_10000.tar")
                input_dim=16
                feature_dim=64
                instance_dim=6
                cluster_dim=10

    # set IK solver
    trac_ik_left = TRACIK(base_link=model_info['base_link'], tip_link=model_info['ee_links'][0], max_time=0.1)
    trac_ik_right = TRACIK(base_link=model_info['base_link'], tip_link=model_info['ee_links'][1], max_time=0.1)

    # set start and goal pose
    if "tocabi" in args.exp_name:
        start_pose_right, start_pose_left = compute_tocabi_grasp_poses(obj_start_pose[:3], obj_start_pose[3:7], condition)
        goal_pose_right, goal_pose_left = compute_tocabi_grasp_poses(obj_goal_pose[:3], obj_goal_pose[3:7], condition)
    else:
        start_pose_right, start_pose_left = compute_dual_grasp_poses(obj_start_pose[:3], obj_start_pose[3:7], condition)
        goal_pose_right, goal_pose_left = compute_dual_grasp_poses(obj_goal_pose[:3], obj_goal_pose[3:7], condition)

    # set start and goal ik group
    start_ik_group = []
    goal_ik_group = []

    # sample start ik group
    start_ik_left = []
    start_ik_right = []
    constraint.planning_scene.add_box('tray', [d1 * 3/4, d1, 0.01], obj_start_pose[:3], obj_start_pose[3:7])
    for _ in range(100):
        joint_seed = np.random.uniform(constraint.lb[:model_info['arm_dofs'][0]], constraint.ub[:model_info['arm_dofs'][0]])
        success_left, ik_left = trac_ik_left.solve(np.array(start_pose_left[0]), np.array(start_pose_left[1]),joint_seed)
        if not success_left:
            continue
        if np.any(ik_left < constraint.lb[model_info['arm_dofs'][0]:]) or np.any(ik_left > constraint.ub[model_info['arm_dofs'][0]:]):
            continue
        start_ik_left.append(ik_left)

    for _ in range(100):
        joint_seed = np.random.uniform(constraint.lb[:model_info['arm_dofs'][0]], constraint.ub[:model_info['arm_dofs'][0]])
        success_right, ik_right = trac_ik_right.solve(np.array(start_pose_right[0]), np.array(start_pose_right[1]), joint_seed)
        if not success_right:
            continue
        if np.any(ik_right < constraint.lb[:model_info['arm_dofs'][0]]) or np.any(ik_right > constraint.ub[:model_info['arm_dofs'][0]]):
            continue
        start_ik_right.append(ik_right)

    for ik_left in start_ik_left:
        for ik_right in start_ik_right:
            start_full_ik = np.concatenate((ik_left, ik_right))
            if constraint.planning_scene.is_valid(start_full_ik):
                start_ik_group.append(start_full_ik)

    # sample goal ik group
    goal_ik_left = []
    goal_ik_right = []
    constraint.planning_scene.add_box('tray', [d1 * 3/4, d1, 0.01], obj_start_pose[:3], obj_start_pose[3:7])
    for _ in range(100):
        joint_seed = np.random.uniform(constraint.lb[:model_info['arm_dofs'][0]], constraint.ub[:model_info['arm_dofs'][0]])
        success_left, ik_left = trac_ik_left.solve(np.array(goal_pose_left[0]), np.array(goal_pose_left[1]),joint_seed)
        if not success_left:
            continue
        if np.any(ik_left < constraint.lb[model_info['arm_dofs'][0]:]) or np.any(ik_left > constraint.ub[model_info['arm_dofs'][0]:]):
            continue
        goal_ik_left.append(ik_left)

    for _ in range(100):
        joint_seed = np.random.uniform(constraint.lb[:model_info['arm_dofs'][0]], constraint.ub[:model_info['arm_dofs'][0]])
        success_right, ik_right = trac_ik_right.solve(np.array(goal_pose_right[0]), np.array(goal_pose_right[1]), joint_seed)
        if not success_right:
            continue
        if np.any(ik_right < constraint.lb[:model_info['arm_dofs'][0]]) or np.any(ik_right > constraint.ub[:model_info['arm_dofs'][0]]):
            continue
        goal_ik_right.append(ik_right)

    for ik_left in goal_ik_left:
        for ik_right in goal_ik_right:
            goal_full_ik = np.concatenate((ik_left, ik_right))
            if constraint.planning_scene.is_valid(goal_full_ik):
                goal_ik_group.append(goal_full_ik)

    return start_ik_group, goal_ik_group, obj_start_pose, obj_goal_pose, scene_dir_local, condition

def select_start_goal_q(start_ik_group, goal_ik_group, metric):
    if metric == 'min_z':
        min_z = np.inf
        min_start_q = None
        min_goal_q = None
        for i in range(len(start_ik_group)):
            for j in range(len(goal_ik_group)):
                z_dist = np.linalg.norm(start_ik_group[i] - goal_ik_group[j])
                if z_dist < min_z:
                    min_z = z_dist
                    min_start_q = start_ik_group[i]
                    min_goal_q = goal_ik_group[j]
        return min_start_q, min_goal_q

    elif metric == 'min_q':
        min_q = np.inf
        min_start_q = None
        min_goal_q = None
        for i in range(len(start_ik_group)):
            for j in range(len(goal_ik_group)):
                q_dist = np.linalg.norm(start_ik_group[i] - goal_ik_group[j])
                if q_dist < min_q:
                    min_start_q = start_ik_group[i]
                    min_goal_q = goal_ik_group[j]
                    min_q = q_dist
        return min_start_q, min_goal_q

    elif metric == 'random':
        min_start_q = start_ik_group[np.random.randint(len(start_ik_group))]
        min_goal_q = goal_ik_group[np.random.randint(len(goal_ik_group))]
        return min_start_q, min_goal_q

    else:
        raise NotImplementedError

def run_planning_trial(args, model_info, constraint, start_q, goal_q):

    if args.method == 'constrained_rrt':
        planner = ConstrainedBiRRT(state_dim=model_info['x_dim'], constraint=constraint)
        planner.set_start(start_q)
        planner.set_goal(goal_q)
        planner.max_distance = model_info['planning']['max_distance_q']
        planner.debug = args.debug
        r, q_path = planner.solve(max_time=args.max_time)
    else:
        raise NotImplementedError

    if r is True:
        print('found a path, planning time', planner.solved_time)
        path_length = np.array([np.linalg.norm(q_path[i + 1] - q_path[i]) for i in range(len(q_path) - 1)]).sum()
        solved_time = planner.solved_time

        cartesian_path = []
        for q in q_path:
            cur_idx = 0
            cartesian_vector = []
            for arm_name, dof in zip(constraint.arm_names, constraint.arm_dofs):
                pos, quat = constraint.forward_kinematics(arm_name, q[cur_idx:cur_idx+dof])
                cur_idx += dof
                cartesian_vector.extend(pos)
                cartesian_vector.extend(quat)
            cartesian_path.append(np.array(cartesian_vector))
        cartesian_path = np.array(cartesian_path)

        if args.display:
            hz = 20
            duration, qs_sample, qds_sample, qdds_sample, ts_sample = time_parameterize(q_path, model_info, hz=hz)
            if args.debug:
                print('duration', duration)
            for q in qs_sample:
                constraint.planning_scene.display(q)
                time.sleep(10.0/hz)
            

    else:
        print('failed to find a path')
        q_path = -1.0
        solved_time = -1.0
        path_length = -1.0
        cartesian_path = -1.0

    return q_path, solved_time, path_length, cartesian_path

if __name__ == '__main__':
    args = parse_args()
    device = args.device
    test_range = range(args.test_scene_start_idx, args.test_scene_start_idx + args.num_test_scenes)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    test_times = []
    test_paths = []
    test_path_lenghts = []
    test_cartesian_paths = []
    test_suc_cnt = 0
    test_cnt = 0
    print(colored(' ---- Start benchmarking ----', 'green'))
    print('exp_name :', args.exp_name)
    print('method   :', args.method)
    print('metric   :', args.metric)

    np.set_printoptions(precision=6, suppress=True)
    print(colored('test_range: ', 'green'), test_range)
    tq = tqdm.tqdm(test_range, position=0)

    for i in tq:
        model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=args.exp_name), 'r'), Loader=yaml.FullLoader)
        if args.exp_name == 'tocabi_orientation':
            constraint = MultiChainFixedOrientationConstraint(arm_names=model_info['arm_names'],
                                                            arm_dofs=model_info['arm_dofs'],
                                                            axis=2,
                                                            base_link=model_info['base_link'],
                                                            ee_links=model_info['ee_links'],
                                                            hand_names=model_info['hand_names'],
                                                            hand_joints=model_info['hand_joints'],
                                                            hand_open=model_info['hand_open'],
                                                            hand_closed=model_info['hand_closed'])
        elif args.exp_name == 'tocabi':
            constraint = MultiChainConstraint(arm_names=model_info['arm_names'],
                                            arm_dofs=model_info['arm_dofs'],
                                            base_link=model_info['base_link'],
                                            ee_links=model_info['ee_links'],
                                            hand_names=model_info['hand_names'],
                                            hand_joints=model_info['hand_joints'],
                                            hand_open=model_info['hand_open'],
                                            hand_closed=model_info['hand_closed'])
        elif args.exp_name == 'ur5_dual_orientation':
            constraint = MultiChainFixedOrientationConstraint(arm_names=model_info['arm_names'],
                                                            arm_dofs=model_info['arm_dofs'],
                                                            axis=2,
                                                            base_link=model_info['base_link'],
                                                            ee_links=model_info['ee_links'],
                                                            hand_names=model_info['hand_names'],
                                                            hand_joints=model_info['hand_joints'],
                                                            hand_open=model_info['hand_open'],
                                                            hand_closed=model_info['hand_closed'])
        elif args.exp_name == 'ur5_dual':
            constraint = MultiChainConstraint(arm_names=model_info['arm_names'],
                                            arm_dofs=model_info['arm_dofs'],
                                            base_link=model_info['base_link'],
                                            ee_links=model_info['ee_links'],
                                            hand_names=model_info['hand_names'],
                                            hand_joints=model_info['hand_joints'],
                                            hand_open=model_info['hand_open'],
                                            hand_closed=model_info['hand_closed'])
        constraint.set_max_iterations(1000)
        constraint.planning_scene.display(np.zeros(16))
        scene_dir = f'dataset/{args.exp_name}/scene_data'
        start_ik_group, goal_ik_group, obj_start_pose, obj_goal_pose, scene_dir_local, condition = load_scene_and_set_start_goal(i, args, scene_dir, constraint)
        start_q, goal_q = select_start_goal_q(start_ik_group, goal_ik_group, args.metric)

        if start_q is None or goal_q is None:
            continue

        d1, d2, theta = condition
        constraint.planning_scene.add_box('tray', [d1 * 3/4, d1, 0.01], obj_start_pose[:3], obj_start_pose[3:7])
        constraint.planning_scene.update_joints(start_q)
        constraint.planning_scene.attach_object('tray', model_info['ee_links'][0], [])

        print("Planning")
        q_path, solved_time, path_length, cartesian_path = run_planning_trial(args, model_info, constraint, start_q, goal_q)
        test_paths.append(q_path)
        test_times.append(solved_time)
        test_path_lenghts.append(path_length)
        test_cartesian_paths.append(cartesian_path)
        test_cnt += 1
        if solved_time > 0:
            test_suc_cnt += 1

        # Compute means of only positive components
        positive_times = [t for t in test_times if t > 0]
        positive_lengths = [l for l in test_path_lenghts if l > 0]
        avg_time = np.mean(positive_times) if positive_times else 0.0
        avg_path_length = np.mean(positive_lengths) if positive_lengths else 0.0

        tq.set_description('test suc rate: {:.3f}, avg time: {:.3f}, avg path length: {:.3f}'.format(
            test_suc_cnt / test_cnt, avg_time, avg_path_length))
    

    # Save the results after each trial
    results = {
        'test_paths': test_paths,
        'test_times': test_times,
        'test_path_lengths': test_path_lenghts,
        'test_cartesian_paths': test_cartesian_paths,
        'test_cnt': test_cnt,
        'test_suc_cnt': test_suc_cnt,
        'avg_time': avg_time,
        'avg_path_length': avg_path_length,
        'args': vars(args),
        'scene_idx': i,
        'scene_dir_local': scene_dir_local,
        'condition': condition,
        'start_q': start_q,
        'goal_q': goal_q,
    }
    save_dir = os.path.join('results', args.exp_name, args.method)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'results_scene_{i:04d}.npz')
    np.savez_compressed(save_path, **results)
        
