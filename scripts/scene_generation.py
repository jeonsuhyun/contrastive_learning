import os
import yaml
import numpy as np
import pickle
import tqdm
import argparse
import time

from srmt.constraints.constraints import MultiChainConstraint, MultiChainFixedOrientationConstraint
from ljcmp.utils.generate_environment import generate_environment
from ljcmp.utils.model_utils import generate_scene_config, load_model
from srmt.kinematics.trac_ik import TRACIK
from scipy.spatial.transform import Rotation as R
from srmt.planning_scene.planning_scene_tools import add_shelf, add_table

def to_python_floats(obj):
    if isinstance(obj, dict):
        return {k: to_python_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_floats(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int64)):
        return float(obj)
    else:
        return obj

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
    T_0o = get_transform(obj_pos, obj_quat)      # Object in world
    T_og = get_transform(obj_to_ee_pos, obj_to_ee_quat)  # object to ee         
    T_0g = T_0o @ T_og                            # EE in world
    return get_pose_from_transform(T_0g)

def compute_dual_grasp_poses(obj_pos, obj_quat, condition=None):
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
    obj_dt_r = (np.pi/2 + theta)
    obj_to_ee_rot_l = np.array([
        [1, 0, 0],
        [0, np.cos(obj_dt_r), -np.sin(obj_dt_r)],
        [0, np.sin(obj_dt_r),  np.cos(obj_dt_r)]
    ])
    obj_to_ee_rot_l =  obj_to_ee_rot_l @ frame_rot
    obj_to_ee_quat_l = R.from_matrix(obj_to_ee_rot_l).as_quat()

    # right arm offset
    obj_to_ee_pos_r = np.array([0.0, -l_obj_y, l_obj_z])
    obj_dt_l = -(np.pi/2 + theta)
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

def generate_random_pose_on_table(table_pos, table_dim, z_height=0.1):
    x_range = (0.35,0.65)
    if table_pos[1] < 0.0:
        y_range = (-0.6,-0.2)
    elif table_pos[1] > 0.0:
        y_range = (0.2,0.6)

    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)
    # print(f"x, y, z_height: {x:.2f}, {y:.2f}, {z_height:.2f}")
    return [float(x), float(y), float(z_height)] + [0.0, 0.0, 0.0, 1.0]


def create_valid_scene(args, scene_id, base_dir, random_seed, constraint, model_info, condition):
    np.random.seed(random_seed)
    d1, d2, theta = condition
    scene = {'c': condition}
    # dim: dphi, dtheta, length, width, height, d)
    # table_top_dim = [length, width, d] x, y d 
    # table_leg_dim = [d, d, height] d, d, z
    table1 = {'pos': [0.6, -0.5, -0.5], 'dim': [0.0, 0.0, 0.5, 0.8, 1.0, 0.05]}
    table2 = {'pos': [0.6,  0.5, -0.5], 'dim': [0.0, 0.0, 0.5, 0.8, 1.0, 0.05]}
    scene['table_1'] = {'pos': table1['pos'], 'dim': table1['dim']}
    scene['table_2'] = {'pos': table2['pos'], 'dim': table2['dim']}

    if args.exp_name == 'tocabi':
        q = np.array([0, -0.3, 1.57, -1.2, -1.57, 1.5, 0.4, -0.2,
                        0, 0.3, -1.57, 1.2, 1.57, -1.5, -0.4, 0.2])
    elif args.exp_name == 'ur5_dual':
        q = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi, 0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0,])
    elif args.exp_name == 'panda_dual':
        q = np.zeros(14)
    else:
        raise ValueError(f"Invalid experiment name: {args.exp_name}")

    add_table(constraint.planning_scene, 'table_1', table1['pos'], 0.0, 0.0, 0.5, 0.8, 1.0, 0.05)
    add_table(constraint.planning_scene, 'table_2', table2['pos'], 0.0, 0.0, 0.5, 0.8, 1.0, 0.05)
    constraint.planning_scene.display(q)
    time.sleep(0.1)

    trac_ik_left = TRACIK(base_link=model_info['base_link'], tip_link=model_info['ee_links'][0], max_time=0.1)
    trac_ik_right = TRACIK(base_link=model_info['base_link'], tip_link=model_info['ee_links'][1], max_time=0.1)

    # ---- Start Pose IK 찾기 ----
    max_pose_attempts = 1000
    print("Generating START pose IKs...")
    for _ in range(max_pose_attempts):
        start_ik_group = []
        start_pose = generate_random_pose_on_table(table_pos=table1['pos'], table_dim=table1['dim'])
        print(f"start_pose: {start_pose}")
        scene['start_pose'] = start_pose
        constraint.planning_scene.add_box('tray', [d1 *3/4, d1, 0.01], start_pose[:3], start_pose[3:])
        start_pose_right, start_pose_left = compute_tocabi_grasp_poses(start_pose[:3], start_pose[3:], condition)

        start_ik_left = []
        for i in range(100):
            joint_seed_left = np.random.uniform(constraint.lb[:model_info['arm_dofs'][0]], constraint.ub[:model_info['arm_dofs'][0]])
            success_left, ik_left = trac_ik_left.solve(np.array(start_pose_left[0]), np.array(start_pose_left[1]), joint_seed_left)
            if not success_left or np.any(ik_left < constraint.lb[model_info['arm_dofs'][0]:]) or np.any(ik_left > constraint.ub[model_info['arm_dofs'][0]:]):
                continue
            start_ik_left.append(ik_left)
        
        if len(start_ik_left) < 8:
            continue

        start_ik_right = []
        for j in range(100):
            joint_seed_right = np.random.uniform(constraint.lb[model_info['arm_dofs'][0]:], constraint.ub[model_info['arm_dofs'][0]:])
            success_right, ik_right = trac_ik_right.solve(np.array(start_pose_right[0]), np.array(start_pose_right[1]), joint_seed_right)
            if not success_right or np.any(ik_right < constraint.lb[:model_info['arm_dofs'][0]]) or np.any(ik_right > constraint.ub[:model_info['arm_dofs'][0]]):
                continue
            start_ik_right.append(ik_right)
        
        if len(start_ik_right) < 8:
            continue

        for ik_left in start_ik_left:
            for ik_right in start_ik_right:
                start_full_ik = np.concatenate((ik_left, ik_right))
                if constraint.planning_scene.is_valid(start_full_ik):
                    start_ik_group.append(start_full_ik)
                    # constraint.planning_scene.display(start_full_ik)
        
        print("start_ik_group: ", len(start_ik_group))

        if len(start_ik_group) > 40:
            break
        
    if len(start_ik_group) < 40:
        return False

    
    # ---- Goal Pose IK 찾기 ----
    print("Generating GOAL pose IKs...")
    for goal_attempt in range(max_pose_attempts):
        goal_ik_group = []
        goal_pose = generate_random_pose_on_table(table_pos=table2['pos'], table_dim=table2['dim'])
        print(f"goal_pose: {goal_pose}")
        scene['goal_pose'] = goal_pose
        constraint.planning_scene.add_box('tray', [d1 *3/4, d1, 0.01], goal_pose[:3], goal_pose[3:])
        goal_pose_right, goal_pose_left = compute_tocabi_grasp_poses(goal_pose[:3], goal_pose[3:], condition)

        goal_ik_left = []
        for _ in range(100):
            joint_seed_left = np.random.uniform(constraint.lb[:model_info['arm_dofs'][0]], constraint.ub[:model_info['arm_dofs'][0]])
            success_left, ik_left = trac_ik_left.solve(np.array(goal_pose_left[0]), np.array(goal_pose_left[1]), joint_seed_left)
            if not success_left or np.any(ik_left < constraint.lb[model_info['arm_dofs'][0]:]) or np.any(ik_left > constraint.ub[model_info['arm_dofs'][0]:]):
                continue
            goal_ik_left.append(ik_left)
        if len(goal_ik_left) < 8:
            continue

        goal_ik_right = []
        for _ in range(100):
            joint_seed_right = np.random.uniform(constraint.lb[model_info['arm_dofs'][0]:], constraint.ub[model_info['arm_dofs'][0]:])
            success_right, ik_right = trac_ik_right.solve(np.array(goal_pose_right[0]), np.array(goal_pose_right[1]), joint_seed_right)
            if not success_right or np.any(ik_right < constraint.lb[:model_info['arm_dofs'][0]]) or np.any(ik_right > constraint.ub[:model_info['arm_dofs'][0]]):
                continue
            goal_ik_right.append(ik_right)
        if len(goal_ik_right) < 8:
            continue

        for ik_left in goal_ik_left:
            for ik_right in goal_ik_right:
                goal_full_ik = np.concatenate((ik_left, ik_right))
                if constraint.planning_scene.is_valid(goal_full_ik):
                    goal_ik_group.append(goal_full_ik)
                    # constraint.planning_scene.display(goal_full_ik)

        if len(goal_ik_group) > 40:
            break
    if len(goal_ik_group) <40:
        return False
    
    scene_dir = os.path.join(base_dir, f'scene_{scene_id:04d}')
    os.makedirs(scene_dir, exist_ok=True)
    with open(os.path.join(scene_dir, 'scene.yaml'), 'w') as f:
        yaml.dump(to_python_floats(scene), f, sort_keys=False)

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='tocabi')
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--total_scenes', type=int, default=100)
    parser.add_argument('--max_scene_attempts', type=int, default=10)
    parser.add_argument('--config_size', type=int, default=100)

    args = parser.parse_args()
    exp_name = args.exp_name

    model_info = yaml.load(open(f'model/{exp_name}/model_info.yaml', 'r'), Loader=yaml.FullLoader)
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

    scene_id = 0
    pbar = tqdm.tqdm(total=args.total_scenes, desc='Generating Valid Scenes')
    condition = [0.3, 0.05, 0.9]
    base_dir = '/home/suhyun/catkin_ws/src/ljcmp/dataset/{}/scene_data'.format(exp_name)
    while scene_id < args.total_scenes:
        print(f"Generating scene {scene_id + 1}/{args.total_scenes}")
        for attempt in range(args.max_scene_attempts):
            success = create_valid_scene(
                args = args,
                scene_id=scene_id,
                random_seed=attempt + scene_id * args.max_scene_attempts,
                base_dir=base_dir,
                constraint=constraint,
                model_info=model_info,
                condition=condition,
            )
            if success:
                pbar.update(1)
                scene_id += 1
                break
    pbar.close()

if __name__ == "__main__":
    main()