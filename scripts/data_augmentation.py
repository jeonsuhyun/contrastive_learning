import argparse
import os
import yaml
import numpy as np
import pickle
import tqdm
import time

from ljcmp.utils.generate_environment import generate_environment
from ljcmp.utils.model_utils import generate_scene_config, load_model
from srmt.kinematics.trac_ik import TRACIK
from scipy.spatial.transform import Rotation as R

def augment_with_ik(dataset_path, save_path, constraint, random_seed=42):
    np.random.seed(random_seed)
    max_ik = 128
    max_attempts = 1000
    # 기존 데이터 로딩
    dataset = np.load(dataset_path)
    joints = dataset[:, 3:]  # 6-DOF 조인트
    new_dataset = []

    print(f"[INFO] Augmenting {len(joints)} joint configurations...")
    trac_ik_left = TRACIK(base_link='base', tip_link='R1_ur5_robotiq_85_gripper', max_time=0.1)
    trac_ik_right = TRACIK(base_link='base', tip_link='R2_ur5_robotiq_85_gripper', max_time=0.1)

    for joint in tqdm.tqdm(joints, desc="Augmenting dataset"):
        pose_right = trac_ik_right.forward_kinematics(joint[:6])
        pose_left = trac_ik_left.forward_kinematics(joint[6:])

        valid_iks = []
        attempts = 0

        while len(valid_iks) < max_ik and attempts < max_attempts:
            # 무작위 시드 생성
            seed = np.random.uniform(constraint.lb, constraint.ub)

            # IK 계산
            success_r, ik_r = trac_ik_right.solve(pose_right[0], pose_right[1], seed[:6])
            success_l, ik_l = trac_ik_left.solve(pose_left[0], pose_left[1], seed[6:])

            if not (success_r and success_l):
                attempts += 1
                continue

            # 범위 검사
            if np.any(ik_r < constraint.lb[:6]) or np.any(ik_r > constraint.ub[:6]):
                attempts += 1
                continue
            if np.any(ik_l < constraint.lb[6:]) or np.any(ik_l > constraint.ub[6:]):
                attempts += 1
                continue

            # 충돌 검사 (옵션)
            ik_full = np.concatenate((ik_r, ik_l))
            if not constraint.planning_scene.is_valid(ik_full):
                attempts += 1
                continue

            # 중복 검사
            is_duplicate = any(np.linalg.norm(ik_full - existing) < 1e-3 for existing in valid_iks)
            if not is_duplicate:
                valid_iks.append(ik_full)
            
            attempts += 1
        # Plot IK results for visualization
        print(len(valid_iks))   
        for ik in valid_iks:
            constraint.planning_scene.display(ik)
            time.sleep(0.1)
            print(np.round(ik[:6], 3), np.round(ik[6:], 3))
        
        import pdb; pdb.set_trace()
        # 결과 저장
        new_dataset.extend(valid_iks)

    import pdb; pdb.set_trace()  # Debugging point to inspect new_dataset
    # 저장
    new_dataset = np.stack(new_dataset)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, new_dataset)
    print(f"[INFO] Augmented dataset saved to {save_path}. Size: {len(new_dataset)}")

    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-E', type=str, default='ur5_dual', 
                        help='panda_orientation, panda_dual, panda_dual_orientation, panda_triple, tocabi, tocabi_orientation, ur5_dual')
    args = parser.parse_args()
    constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(args.exp_name)

    new_dataset = augment_with_ik(
        dataset_path=f'dataset/{args.exp_name}/manifold/data_fixed_50000.npy',
        save_path=f'dataset/{args.exp_name}/manifold/data_fixed_50000_augmented.npy',
        constraint=constraint,
        random_seed=42
    )