import os
import yaml
import numpy as np
import pickle
import tqdm

from ljcmp.utils.generate_environment import generate_environment
from ljcmp.utils.model_utils import generate_scene_config, load_model
from srmt.kinematics.trac_ik import TRACIK
from scipy.spatial.transform import Rotation as R

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

def compute_dual_grasp_poses(obj_pos, obj_quat, condition = None):
    if condition is None:
        condition = [0.3, 0.05, 0.9]
    # R2(right) 손 offset
    d1 = condition[0]
    d2 = condition[1]
    theta = condition[2]

    l_obj_z = d2 * np.sin(theta)
    l_obj_y = d1/2 + d2 * np.cos(theta)
    obj_to_ee_pos_r = np.array([0.0, l_obj_y, l_obj_z])
    obj_dt_r = (np.pi/2 + theta)
    obj_to_ee_rot_r = np.array([
        [1, 0, 0],
        [0, np.cos(obj_dt_r), -np.sin(obj_dt_r)],
        [0, np.sin(obj_dt_r),  np.cos(obj_dt_r)]
    ])
    obj_to_ee_quat_r = R.from_matrix(obj_to_ee_rot_r).as_quat()

    # R1(left) 손 offset
    obj_to_ee_pos_l = np.array([0.0, -l_obj_y, l_obj_z])
    obj_dt_l = -(np.pi/2 + theta)
    obj_to_ee_rot_l = np.array([
        [1, 0, 0],
        [0, np.cos(obj_dt_l), -np.sin(obj_dt_l)],
        [0, np.sin(obj_dt_l),  np.cos(obj_dt_l)]
    ])
    obj_to_ee_quat_l = R.from_matrix(obj_to_ee_rot_l).as_quat()

    # grasp pose 계산
    r_pos, r_quat = compute_grasp_pose(obj_pos, obj_quat, obj_to_ee_pos_r, obj_to_ee_quat_r)
    l_pos, l_quat = compute_grasp_pose(obj_pos, obj_quat, obj_to_ee_pos_l, obj_to_ee_quat_l)

    return (r_pos, r_quat), (l_pos, l_quat)

def generate_random_pose_on_table(table_pos, table_dim, z_height=0.65):
    x_range = (table_pos[0] - table_dim[2]/2 + 0.05, table_pos[0] + table_dim[2]/2 -0.05)

    if table_pos[1] < 0.0:
        y_range = (table_pos[1], table_pos[1] + table_dim[3]/2 - 0.05)
    elif table_pos[1] > 0.0:
        y_range = (table_pos[1] - table_dim[3]/2 + 0.05, table_pos[1])

    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)
    # print(f"x, y, z_height: {x:.2f}, {y:.2f}, {z_height:.2f}")
    return [float(x), float(y), float(z_height)] + [0.0, 0.0, 0.0, 1.0]


def create_valid_scene(scene_id, base_dir, random_seed, constraint, constraint_model, update_scene_from_yaml, model_info, condition, config_size=100, max_attempts=500):
    np.random.seed(random_seed)
    scene = {'c': [0.3, 0.05, 0.9]}
    # dim: dphi, dtheta, length, width, height, d)
    # table_top_dim = [length, width, d] 
    # table_leg_dim = [d, d, height] 
    table1 = {'pos': [0.5, -0.6, 0.3], 'dim': [0.0, 0.0, 0.5, 0.9, 0.6, 0.05]}
    table2 = {'pos': [0.5,  0.6, 0.3], 'dim': [0.0, 0.0, 0.5, 0.9, 0.6, 0.05]}
    scene['table_1'] = {'pos': table1['pos'], 'dim': table1['dim']}
    scene['table_2'] = {'pos': table2['pos'], 'dim': table2['dim']}

    obs_boxes = []
    obs = {'pos': [0.5, 0.0, 0.35], 'dim': [0.5, 0.05, 0.7]}
    obs_boxes.append(obs)
    scene['obs0'] = {'pos': [0.5, 0.0, 0.35], 'dim': [0.5, 0.05, 0.7]}

    constraint.planning_scene.add_box('obs0', obs['dim'], obs['pos'], [1,0,0,0])
    q = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi, 0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0,])
    constraint.planning_scene.display(q)

    trac_ik_left = TRACIK(base_link='base', tip_link='R1_ur5_robotiq_85_gripper', max_time=0.1)
    trac_ik_right = TRACIK(base_link='base', tip_link='R2_ur5_robotiq_85_gripper', max_time=0.1)

    # ---- Start Pose IK 찾기 ----
    max_pose_attempts = 10000
    start_ik_group = []
    print("Generating START pose IKs...")
    for start_attempt in range(max_pose_attempts):
        start_pose = generate_random_pose_on_table(table_pos=table1['pos'], table_dim=table1['dim'])
        scene['start_pose'] = start_pose
        
        start_pose_right, start_pose_left = compute_dual_grasp_poses(start_pose[:3], start_pose[3:], condition)
        start_ik_group.clear()

        for _ in range(100):
            joint_seed = np.random.uniform(constraint.lb[:6], constraint.ub[:6])
            success_left, ik_left = trac_ik_left.solve(np.array(start_pose_left[0]), np.array(start_pose_left[1]), joint_seed)
            if not success_left or np.any(ik_left < constraint.lb[6:]) or np.any(ik_left > constraint.ub[6:]):
                continue
            success_right, ik_right = trac_ik_right.solve(np.array(start_pose_right[0]), np.array(start_pose_right[1]), joint_seed)
            if not success_right or np.any(ik_right < constraint.lb[:6]) or np.any(ik_right > constraint.ub[:6]):
                continue
            start_full_ik = np.concatenate((ik_right, ik_left))
            if constraint.planning_scene.is_valid(start_full_ik):
                start_ik_group.append(start_full_ik)
                constraint.planning_scene.display(start_full_ik)

        if len(start_ik_group) > 40:
            break
    
    # ---- Goal Pose IK 찾기 ----
    print("Generating GOAL pose IKs...")
    goal_ik_group = []
    for goal_attempt in range(max_pose_attempts):
        goal_pose = generate_random_pose_on_table(table_pos=table2['pos'], table_dim=table2['dim'])
        scene['goal_pose'] = goal_pose
        
        goal_pose_right, goal_pose_left = compute_dual_grasp_poses(goal_pose[:3], goal_pose[3:], condition)
        goal_ik_group.clear()

        for _ in range(100):
            joint_seed = np.random.uniform(constraint.lb[:6], constraint.ub[:6])
            success_left, ik_left = trac_ik_left.solve(np.array(goal_pose_left[0]), np.array(goal_pose_left[1]), joint_seed)
            if not success_left or np.any(ik_left < constraint.lb[6:]) or np.any(ik_left > constraint.ub[6:]):
                continue
            success_right, ik_right = trac_ik_right.solve(np.array(goal_pose_right[0]), np.array(goal_pose_right[1]), joint_seed)
            if not success_right or np.any(ik_right < constraint.lb[:6]) or np.any(ik_right > constraint.ub[:6]):
                continue
            goal_full_ik = np.concatenate((ik_right, ik_left))
            
            if constraint.planning_scene.is_valid(goal_full_ik):
                goal_ik_group.append(goal_full_ik)
                constraint.planning_scene.display(goal_full_ik)
            
        if len(goal_ik_group) > 40:
            break
    
    scene_dir = os.path.join(base_dir, f'scene_{scene_id:04d}')
    os.makedirs(scene_dir, exist_ok=True)
    with open(os.path.join(scene_dir, 'scene.yaml'), 'w') as f:
        yaml.dump(to_python_floats(scene), f, sort_keys=False)

    return True

def generate_valid_scenes(total_scenes=100,
                          base_dir='/home/suhyun/catkin_ws/src/ljcmp/dataset/ur5_dual/scene_data',
                          config_size=100, max_scene_attempts=100,
                          constraint=None, constraint_model=None,
                          update_scene_from_yaml=None,
                          model_info=None, condition=None):
    scene_id = 0
    pbar = tqdm.tqdm(total=total_scenes, desc='Generating Valid Scenes')
    
    while scene_id < total_scenes:
        print(f"Generating scene {scene_id + 1}/{total_scenes}")
        for attempt in range(max_scene_attempts):
            success = create_valid_scene(
                scene_id=scene_id,
                random_seed=attempt + scene_id * max_scene_attempts,
                base_dir=base_dir,
                constraint=constraint,
                constraint_model=constraint_model,
                update_scene_from_yaml=update_scene_from_yaml,
                model_info=model_info,
                condition=condition,
                config_size=config_size
            )
            if success:
                pbar.update(1)
                scene_id += 1
                break
    pbar.close()

exp_name = 'ur5_dual'
# exp_name = 'ur5_dual_orientation'
# exp_name = 'tocabi'
# exp_name = 'tocabi_orientation'

constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(exp_name)
constraint.set_early_stopping(True)
constraint_model, _ = load_model(exp_name, model_info, False)

generate_valid_scenes(
    total_scenes=100,
    base_dir='/home/suhyun/catkin_ws/src/ljcmp/dataset/{}/scene_data'.format(exp_name),
    config_size=100,
    max_scene_attempts=10,
    constraint=constraint,
    constraint_model=constraint_model,
    update_scene_from_yaml=update_scene_from_yaml,
    model_info=model_info,
    condition=condition
)