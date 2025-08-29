#!/usr/bin/env python3

import numpy as np
import yaml
import argparse
import time
import tqdm
import pickle

from scipy.linalg import null_space
from math import pi
from scipy.spatial.transform import Rotation as R
from srmt.kinematics.trac_ik import TRACIK
from srmt.constraints.constraints import MultiChainConstraint
from ljcmp.utils.model_utils import get_transform, get_pose_from_transform
from ljcmp.utils.generate_environment import generate_environment


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

def solve_ik_for_pose(arm_name, pc, pose_pos, pose_quat, trac_ik, constraint_lb, constraint_ub, max_attempts=100):
    """Solve 8 IK solutions for a single pose with multiple attempts"""
    solutions = []
    
    for attempt in range(max_attempts):
        # Random seed for IK
        joint_seed = np.random.uniform(constraint_lb, constraint_ub)
        success, ik_solution = trac_ik.solve(pose_pos, pose_quat, joint_seed)
        if success:
            if np.any(ik_solution < constraint_lb) or np.any(ik_solution > constraint_ub):
                continue
            if arm_name == "left":
                ik_concat = np.concatenate([ik_solution, len(constraint_lb)* [0]])
            elif arm_name == "right":
                ik_concat = np.concatenate([len(constraint_lb) * [0], ik_solution])
            
            else:
                raise ValueError(f"Invalid arm name: {arm_name}")

            if not pc.is_valid(ik_concat):
                continue

            is_unique = True
            for existing_sol in solutions:
                if np.linalg.norm(ik_solution - existing_sol) < 0.01:
                    is_unique = False
                    break
            
            if is_unique:
                solutions.append(ik_solution)
        if len(solutions) == 8:
            return solutions

    return solutions

def main():
    parser = argparse.ArgumentParser(description='Solve IK for object pose with dual arms')
    parser.add_argument('--exp_name', type=str, default='tocabi', 
                       help='Robot configuration name (tocabi, ur5_dual, panda_dual, panda_orientation)')
    parser.add_argument('--obj_pos', nargs=3, type=float, default=[0.5, 0.0, 0.1], 
                       help='Object position [x, y, z]')
    parser.add_argument('--obj_quat', nargs=4, type=float, default=[0.0, 0.0, 0.0, 1.0], 
                       help='Object quaternion [x, y, z, w]')
    parser.add_argument('--condition', nargs=3, type=float, default=[0.3, 0.1, 0.9], 
                       help='Grasp condition [d1, d2, theta]')
    parser.add_argument('--max_attempts', type=int, default=50, 
                       help='Maximum IK attempts per arm')
    parser.add_argument('--max_iter', type=int, default=500, 
                       help='Maximum iterations for IK')
    parser.add_argument('--display', action='store_true', 
                       help='Display solutions in planning scene')
    parser.add_argument('--num_positions', type=int, default=1000,
                       help='Number of object positions to generate')
    parser.add_argument('--pos_range_x', nargs=2, type=float, default=[0.2, 0.8],
                       help='Range for x position [min, max]')
    parser.add_argument('--pos_range_y', nargs=2, type=float, default=[-0.6, 0.6],
                       help='Range for y position [min, max]')
    parser.add_argument('--pos_range_z', nargs=2, type=float, default=[0.0, 0.8],
                       help='Range for z position [min, max]')
    parser.add_argument('--rot_range_z', nargs=2, type=float, default=[-np.pi/2, np.pi/2],
                       help='Range for rotation z [min, max]')
    parser.add_argument('--rot_range_x', nargs=2, type=float, default=[-np.pi/2, np.pi/2],
                       help='Range for rotation x [min, max]')
    parser.add_argument('--rot_range_y', nargs=2, type=float, default=[-np.pi/2, np.pi/2],
                       help='Range for rotation y [min, max]')
    parser.add_argument('--rotation_mode', type=str, default='z_only', 
                       choices=['z_only', 'xyz_constrained', 'random_3d'],
                       help='Rotation mode for object orientation')
    args = parser.parse_args()

    # Load model info    
    model_info = yaml.load(open(f'model/{args.exp_name}/model_info.yaml', 'r'), Loader=yaml.FullLoader)

    print(f"Robot: {args.exp_name}")
    print(f"Grasp condition: {args.condition}")
    print("-" * 50)
    
    # Initialize TRACIK solvers
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
    pc.display(np.array([0, -0.3, 1.57, -1.2, -1.57, 1.5, 0.4, -0.2,
                        0, 0.3, -1.57, 1.2, 1.57, -1.5, -0.4, 0.2]))
    time.sleep(1)
    trac_ik_left = TRACIK(base_link=model_info['base_link'], tip_link=model_info['ee_links'][0], max_time=0.1)
    trac_ik_right = TRACIK(base_link=model_info['base_link'], tip_link=model_info['ee_links'][1], max_time=0.1)

    # Initialize results storage
    results_data = {
        'robot_config': args.exp_name,
        'generation_parameters': {
            'num_positions': args.num_positions,
            'pos_range_x': args.pos_range_x,
            'pos_range_y': args.pos_range_y,
            'pos_range_z': args.pos_range_z,
            'rot_range_x': args.rot_range_x,
            'rot_range_y': args.rot_range_y,
            'rot_range_z': args.rot_range_z,
            'rotation_mode': args.rotation_mode,
            'condition': args.condition,
            'max_attempts': args.max_attempts
        },
        'joints': [],
        'jacobians': [],
        'nulls': []
    }
    # Process each object position
    print(f"\nProcessing {args.num_positions} object positions...")
    success_count = 0
    while success_count < args.num_positions:
        combined_solutions = []
        jacobians = []
        nulls = []
        
        # Generate random position
        x = np.random.uniform(args.pos_range_x[0], args.pos_range_x[1])
        y = np.random.uniform(args.pos_range_y[0], args.pos_range_y[1])
        z = np.random.uniform(args.pos_range_z[0], args.pos_range_z[1])
        obj_pos = [x, y, z]
        
        # Generate random quaternion for object orientation based on rotation mode
        if args.rotation_mode == 'z_only':
            # Random rotation around Z-axis only
            z_angle = np.random.uniform(args.rot_range_z[0], args.rot_range_z[1])
            obj_quat = R.from_euler('z', z_angle).as_quat()
            
        elif args.rotation_mode == 'xyz_constrained':
            # Random rotation with constraints on all axes
            x_angle = np.random.uniform(args.rot_range_x[0], args.rot_range_x[1])
            y_angle = np.random.uniform(args.rot_range_y[0], args.rot_range_y[1])
            z_angle = np.random.uniform(args.rot_range_z[0], args.rot_range_z[1])
            obj_quat = R.from_euler('xyz', [x_angle, y_angle, z_angle]).as_quat()
            
        elif args.rotation_mode == 'random_3d':
            # Completely random 3D rotation
            obj_quat = R.random().as_quat()

        print(f"\n{'='*60}")
        print(f"Processing object position {success_count + 1}/{args.num_positions}: {obj_pos}", f"Quaternion: {obj_quat}")
        print(f"{'='*60}")
        

        obj_pos = np.array(obj_pos)
        obj_quat = np.array(obj_quat)  # Use the randomly generated quaternion
        condition = np.array(args.condition)
        d1, d2, theta = condition
        pc.add_box('tray', [d1 *3/4, d1, 0.01], obj_pos, obj_quat)
        print("Computing grasp poses...")
        # Dual arm robot
        right_grasp_pose, left_grasp_pose = compute_tocabi_grasp_poses(obj_pos, obj_quat, condition)
    
        # Solve IK for arms
        print("Solving IK for arms...")
        right_solutions = solve_ik_for_pose("right",
            pc, right_grasp_pose[0], right_grasp_pose[1], 
            trac_ik_right, constraint.lb[:model_info['arm_dofs'][0]], constraint.ub[:model_info['arm_dofs'][0]], args.max_attempts
        )
        left_solutions = solve_ik_for_pose("left",
            pc, left_grasp_pose[0], left_grasp_pose[1], 
            trac_ik_left, constraint.lb[model_info['arm_dofs'][0]:], constraint.ub[model_info['arm_dofs'][0]:], args.max_attempts
        )
        
        if len(right_solutions) < 8 or len(left_solutions) < 8:
            continue
        
        for right_sol in right_solutions:
            for left_sol in left_solutions:
                combined_solution = np.concatenate([left_sol, right_sol])
                if not pc.is_valid(combined_solution):
                    continue
                if args.display:
                    pc.display(combined_solution)
                    time.sleep(0.1)
                combined_solutions.append(combined_solution)
                jac = constraint.jacobian(combined_solution)
                jacobians.append(jac)
                nulls.append(null_space(jac))

        print(f"Total combined solutions: {len(combined_solutions)}")
        if len(combined_solutions) < 50:
            continue        
        # Store results for this position
        position_data = {
            'joints': [sol.tolist() for sol in combined_solutions],
            'jacobians': [jac.tolist() for jac in jacobians],
            'nulls': [null.tolist() for null in nulls]
        }
        
        results_data['joints'].append(position_data['joints'])
        results_data['jacobians'].append(position_data['jacobians'])
        results_data['nulls'].append(position_data['nulls'])
        success_count += 1
        print(f"Completed position {success_count + 1}/{args.num_positions}")
    

    # Save to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'ik_results_{args.exp_name}_{args.rotation_mode}_{timestamp}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"All results saved to {filename}")
    print(f"Total positions processed: {args.num_positions}")


if __name__ == "__main__":
    main() 