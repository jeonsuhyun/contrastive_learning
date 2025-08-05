
import torch
import os
import yaml
import tqdm
import copy
import time
import numpy as np
import pickle
import networkx as nx

import matplotlib.pyplot as plt
from ljcmp.models import TSVAE
from ljcmp.models.validity_network import VoxelValidityNet
from contrastiveik.modules import resnet, network, transform
from ljcmp.planning.sample_region import RegionSampler, LatentRegionSampler

from ljcmp.planning.constrained_bi_rrt import SampleBiasedConstrainedBiRRT
from ljcmp.planning.precomputed_roadmap import PrecomputedRoadmap, PrecomputedGraph
from ljcmp.planning.constrained_bi_rrt_latent_jump import ConstrainedLatentBiRRT
from ljcmp.planning.constrained_bi_rrt import ConstrainedBiRRT
from ljcmp.utils.time_parameterization import time_parameterize

from scipy.linalg import null_space
from scipy.spatial.transform import Rotation as R
from srmt.kinematics.trac_ik import TRACIK
from termcolor import colored

import multiprocessing as mp

def load_model(exp_name, model_info, load_validity_model=False):
    constraint_model_path = 'model/{exp_name}/{model_path}'.format(exp_name=exp_name, 
                                                                    model_path=model_info['constraint_model']['path'])
    
    model_type = constraint_model_path.split('.')[-1]
    tag = model_info['constraint_model']['tag']

    if model_type == 'pt':
        constraint_model = torch.load(constraint_model_path)

    elif model_type == 'ckpt':
        constraint_model_checkpoint = torch.load(constraint_model_path)
        constraint_model_state_dict = constraint_model_checkpoint['state_dict']
        constraint_model = TSVAE(x_dim=model_info['x_dim'], 
                                 h_dim=model_info['constraint_model']['h_dim'], 
                                 z_dim=model_info['z_dim'], 
                                 c_dim=model_info['c_dim'], 
                                 null_augment=False)
        
        for key in list(constraint_model_state_dict):
            constraint_model_state_dict[key.replace("model.", "")] = constraint_model_state_dict.pop(key)

        constraint_model.load_state_dict(constraint_model_state_dict)
        # save pt
        os.makedirs('model/{exp_name}/weights/{tag}'.format(exp_name=exp_name, tag=tag), exist_ok=True)
        torch.save(constraint_model, 'model/{exp_name}/weights/{tag}/constraint_model.pt'.format(exp_name=exp_name, tag=tag))

    else:
        raise NotImplementedError
    
    constraint_model.eval()

    tag = model_info['voxel_validity_model']['tag']
    validity_model_path = 'model/{exp_name}/{model_path}'.format(exp_name=exp_name, 
                                                                  model_path=model_info['voxel_validity_model']['path'])
    validity_model = VoxelValidityNet(z_dim=model_info['z_dim'], 
                                      c_dim=model_info['c_dim'], 
                                      h_dim=model_info['voxel_validity_model']['h_dim'],
                                      voxel_latent_dim=model_info['voxel_validity_model']['voxel_latent_dim'])

    if load_validity_model:
        validity_model_type = validity_model_path.split('.')[-1]
        if validity_model_type == 'pt':
            validity_model = torch.load(validity_model_path)

        elif validity_model_type == 'ckpt':
            validity_model_checkpoint = torch.load(validity_model_path)
            validity_model_state_dict = validity_model_checkpoint['state_dict']
            validity_model_state_dict_z_model = {}
            for key in list(validity_model_state_dict):
                if key.startswith('model.'):
                    validity_model_state_dict_z_model[key.replace("model.", "")] = validity_model_state_dict.pop(key)

            validity_model.load_state_dict(validity_model_state_dict_z_model)
            # save pt
            os.makedirs('model/{exp_name}/weights/{tag}'.format(exp_name=exp_name, tag=tag), exist_ok=True)
            torch.save(validity_model, 'model/{exp_name}/weights/{tag}/voxel_validity_model.pt'.format(exp_name=exp_name, tag=tag))

        else:
            raise NotImplementedError

        validity_model.threshold = model_info['voxel_validity_model']['threshold']
    else:
        validity_model.threshold = 0.0

    validity_model.eval()

    return constraint_model, validity_model


def generate_constrained_config(constraint_setup_fn, exp_name,
                                workers_seed_range=range(0,2), dataset_size=30000, 
                                samples_per_condition=10,
                                save_top_k=1, save_every=1000, timeout=1.0,
                                fixed_condition=None,
                                display=False, display_delay=0.5):
    save_dir = f'dataset/{exp_name}/manifold/'
    model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=exp_name), 'r'), Loader=yaml.FullLoader)

    dataset_size_per_worker = dataset_size // len(workers_seed_range)

    def generate_constrained_config_worker(seed, pos):
        np.random.seed(seed)
        save_dir_local = os.path.join(save_dir, str(seed))
        os.makedirs(save_dir_local, exist_ok=True)

        tq = tqdm.tqdm(total=dataset_size_per_worker, position=pos,desc='Generating dataset for seed {}'.format(seed))
        q_dataset = []
        jac_dataset = []
        null_dataset = []
        
        constraint, set_constraint_by_condition = constraint_setup_fn()
        pc = constraint.planning_scene
        
        ###### debug #######
        # if display:
        #     q = np.array([0, 0, 0, 0, 0, 0, 0, 0,
        #                   0, 0, 0, 0, 0, 0, 0, 0])
        #     pc.display(q)
        #     time.sleep(5)
        ####################
        if fixed_condition is not None:
            set_constraint_by_condition(fixed_condition)
            c = fixed_condition

        print('dataset_size_per_worker', dataset_size_per_worker)
        print('len(q_dataset)', len(q_dataset))
        while len(q_dataset) < dataset_size_per_worker:
            if fixed_condition is None:
                c = np.random.uniform(model_info['c_lb'], model_info['c_ub'])
                set_constraint_by_condition(c)

            q = constraint.sample_valid(pc.is_valid, timeout=timeout)
            if q is False:
                continue
            # print('aa')
            # if display:
            #     pc.display(q)
            #     time.sleep(display_delay)

            if (q > constraint.ub).any() or (q < constraint.lb).any():
                continue
            
            jac = constraint.jacobian(q)
            null = null_space(jac)
            q_dataset.append(np.concatenate((c,q)))
            jac_dataset.append(jac)
            null_dataset.append(null)
            
            tq.update(1)

            for _ in range(samples_per_condition-1):
                q = constraint.sample_valid(pc.is_valid, timeout=timeout)

                if q is False:
                    continue

                if (q > constraint.ub).any() or (q < constraint.lb).any():
                    continue

                jac = constraint.jacobian(q)
                null = null_space(jac)
                q_dataset.append(np.concatenate((c,q)))
                jac_dataset.append(jac)
                null_dataset.append(null)
                
                tq.update(1)

                if display:
                    pc.display(q)
                    time.sleep(display_delay)
                
                if save_every > 0:
                    if len(q_dataset) % save_every == 0:
                        current_len = len(q_dataset)
                        delete_len = current_len - save_every*save_top_k
                        try:
                            np.save(f'{save_dir_local}/data_{seed}_{current_len}.npy', np.array(q_dataset))
                            np.save(f'{save_dir_local}/null_{seed}_{current_len}.npy', np.array(null_dataset))
                            
                            if delete_len > 0:
                                os.remove(f'{save_dir_local}/data_{seed}_{delete_len}.npy')
                                os.remove(f'{save_dir_local}/null_{seed}_{delete_len}.npy')
                        except:
                            print('save failed')
                    break

        np.save(f'{save_dir_local}/data_{seed}_{dataset_size_per_worker}.npy', np.array(q_dataset[:dataset_size_per_worker]))
        np.save(f'{save_dir_local}/null_{seed}_{dataset_size_per_worker}.npy', np.array(null_dataset[:dataset_size_per_worker]))
        tq.close()

    # generate_constrained_config_worker(1107,0)
    p_list = []
    for pos, seed in enumerate(workers_seed_range):
        p = mp.Process(target=generate_constrained_config_worker, args=(seed, pos))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    print('Merge dataset')
    data_list = []
    null_list = []
    for seed in workers_seed_range:
        save_dir_local = os.path.join(save_dir, str(seed))
        data_list.append(np.load(f'{save_dir_local}/data_{seed}_{dataset_size_per_worker}.npy'))
        null_list.append(np.load(f'{save_dir_local}/null_{seed}_{dataset_size_per_worker}.npy'))
    
    data = np.concatenate(data_list)
    null = np.concatenate(null_list)

    if fixed_condition is not None:
        np.save(f'{save_dir}/data_fixed_{dataset_size}.npy', data)
        np.save(f'{save_dir}/null_fixed_{dataset_size}.npy', null)
    else:
        np.save(f'{save_dir}/data_{dataset_size}.npy', data)
        np.save(f'{save_dir}/null_{dataset_size}.npy', null)



    print('Done')

def generate_scene_config(constraint, constraint_model, model_info, condition, update_scene_from_yaml, start=0, end=500, config_size=100):
    save_dir = f"dataset/{model_info['name']}/scene_data"
    os.makedirs(save_dir, exist_ok=True)

    tq_scene = tqdm.tqdm(range(start, end))
    for cnt in tq_scene:
        tq_scene.set_description('scene: {:04d}'.format(cnt))
        save_dir_local = '{}/scene_{:04d}'.format(save_dir, cnt)
        if os.path.exists(f'{save_dir_local}/scene.yaml'):
            
            scene_data = yaml.load(open(f'{save_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
            update_scene_from_yaml(scene_data)
            invalid_by_projection = 0
            invalid_by_out_of_range = 0
            invalid_by_collision = 0
            valid_sets = []
            invalid_sets = []
            tq = tqdm.tqdm(total=config_size, leave=False)
            
            while len(valid_sets) < config_size or len(invalid_sets) < config_size:
                xs, zs = constraint_model.sample(100)    
                for x, z in zip(xs, zs):
                    r = constraint.project(x)

                    if r is False:
                        if len(invalid_sets) < config_size:
                            invalid_sets.append((z,x,condition))
                        invalid_by_projection += 1
                        continue
                    
                    if (x < constraint.lb).any() or (x > constraint.ub).any():
                        if len(invalid_sets) < config_size:
                            invalid_sets.append((z,x,condition))
                        invalid_by_out_of_range += 1
                        continue

                    r = constraint.planning_scene.is_valid(x)
                    if r:
                        if len(valid_sets) < config_size:
                            valid_sets.append((z, x, condition))
                            tq.update(1)
                    else:
                        if len(invalid_sets) < config_size:
                            invalid_sets.append((z, x, condition))
                        invalid_by_collision += 1
                    
                    tq.set_description(f'valid: {len(valid_sets)}, invalid: {len(invalid_sets)} (P: {invalid_by_projection}, R: {invalid_by_out_of_range}, C: {invalid_by_collision})')
            tq.close()
            save_dir_local_tag = '{}/{}'.format(save_dir_local, model_info['constraint_model']['tag'])
            os.makedirs(save_dir_local_tag, exist_ok=True)
            pickle.dump({'valid_set':valid_sets, 'invalid_set':invalid_sets}, open(f'{save_dir_local_tag}/config.pkl', 'wb'))
                    
        else:
            print('scene {} not found'.format(cnt))
            break
            


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

def plot_joint_pairwise_scatter(goal_ik_group):
    goal_ik_group = np.array(goal_ik_group)
    N, k = goal_ik_group.shape
    fig, axes = plt.subplots(k, k, figsize=(2 * k, 2 * k))
    fig.suptitle("Pairwise Joint Scatter Plots", fontsize=16)

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]

            if i == j:
                # 대각선: 1D histogram
                ax.hist(goal_ik_group[:, i], bins=30, color='gray', alpha=0.7)
            else:
                # 하위 삼각: scatter plot
                ax.scatter(goal_ik_group[:, j], goal_ik_group[:, i], s=5, alpha=0.6)

            if i == k - 1:
                ax.set_xlabel(f'J{j+1}')
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(f'J{i+1}')
            else:
                ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

# 사용 예시
# plot_joint_pairwise_scatter(goal_ik_group)

def benchmark(args, exp_name, model_info, method, update_scene_from_yaml, 
              constraint, device='cpu', condition=None, max_time=500.0,
              use_given_start_goal=False, debug=False, display=False,
              load_validity_model=True, 
              trials=10, test_scene_start_idx=500, num_test_scenes=100):
    """_summary_

    Args:
        exp_name (_type_): _description_
        model_info (_type_): _description_
        method (str): method name (e.g. 'latent_rrt', 'latent_rrt_latent_jump', 'sampling_rrt')
        update_scene_from_yaml (_type_): _description_
        constraint (_type_): _description_
        use_given_start_goal (bool, optional): _description_. Defaults to False.
        test_scene_start_idx (int, optional): _description_. Defaults to 500.
        num_test_scenes (int, optional): _description_. Defaults to 100.
    """
    # ready for model

    if "latent" in method:
        constraint_model, validity_model = load_model(exp_name, model_info, 
                                                    load_validity_model=load_validity_model)

        constraint_model.to(device=device)
        validity_model.to(device=device)

        if condition is not None:
            constraint_model.set_condition(condition)
            validity_model.set_condition(condition)

        # warm up
        z_dim = model_info['z_dim']
        z = torch.normal(mean=torch.zeros([constraint_model.default_batch_size, z_dim]), 
                        std=torch.ones([constraint_model.default_batch_size, z_dim])).to(device=device)
        _ = validity_model(z)
    
    else:
        constraint_model = None
        validity_model = None


    if 'precomputed_roadmap' in method:
        tag = model_info['precomputed_roadmap']['tag']

        precomputed_roadmap_path = os.path.join('model', 
                                                exp_name, 
                                                model_info['precomputed_roadmap']['path'])
        
        precomputed_roadmap = nx.read_gpickle(precomputed_roadmap_path)

        print(colored('precomputed_roadmap tag: ', 'green'), tag)
        print(colored('precomputed_roadmap path: ', 'green'), precomputed_roadmap_path)
        print(colored('precomputed_roadmap nodes: ', 'green'), len(precomputed_roadmap.nodes))
        print(colored('precomputed_roadmap edges: ', 'green'), len(precomputed_roadmap.edges))

    if 'precomputed_graph' in method:
        tag = model_info['precomputed_graph']['tag']

        precomputed_graph_path = os.path.join('dataset',
                                               exp_name,
                                               model_info['precomputed_graph']['path'])
        
        configs = np.load(precomputed_graph_path)
        
        planner = PrecomputedGraph(state_dim=model_info['x_dim'], constraint=constraint)
        planner.from_configs(configs[:, model_info['c_dim']:])
        
        precomputed_graph = planner.graph

    # benchmark
    scene_dir = f'dataset/{exp_name}/scene_data'

    test_range = range(test_scene_start_idx, test_scene_start_idx + num_test_scenes)
    
    test_times = []
    test_paths = []
    test_path_lenghts = []
    test_paths_z = [] # only for latent_rrt
    test_path_refs = [] # only for latent_rrt
    test_suc_cnt = 0
    test_cnt = 0

    print(colored('test_range: ', 'green'), test_range)
    tq = tqdm.tqdm(test_range, position=0)
    for i in tq:
        scene_dir_local = '{}/scene_{:04d}'.format(scene_dir, i)
        if not os.path.exists(f'{scene_dir_local}/scene.yaml'):
            print(f'{scene_dir_local}/scene.yaml not exist')
            break
        
        scene = yaml.load(open(os.path.join(scene_dir_local,  'scene.yaml'), 'r'), Loader=yaml.FullLoader)
        update_scene_from_yaml(scene)

        if "panda" in exp_name:
            given_start_q = np.loadtxt(os.path.join(scene_dir_local, 'start_q.txt'))
            given_goal_q = np.loadtxt(os.path.join(scene_dir_local, 'goal_q.txt'))
        
        obj_start_pose = np.array(scene['start_pose'])
        obj_goal_pose = np.array(scene['goal_pose'])

        if validity_model is not None:
            voxel = np.load(os.path.join(scene_dir_local, 'voxel.npy')).flatten()
            validity_model.set_voxel(voxel)

        ###### debug #######
        if args.metric == 'given':
            feature_map = False
        else:
            feature_map = True

        if feature_map:
            # print("metric: ", args.metric)
            
            if "panda_dual" in exp_name:
                if "panda_dual_orientation" in exp_name:
                    model_path = os.path.join('contrastiveik','save',"panda_dual_orientation_fixed", "checkpoint_4000.tar")
                    input_dim=14
                    feature_dim=64 
                    instance_dim=6
                    cluster_dim=8
                else:
                    # model_path = os.path.join('contrastiveik','save',"panda_dual_fixed_0417", "checkpoint_400.tar")
                    model_path = os.path.join('contrastiveik','save',"panda_dual_fixed", "checkpoint_10000.tar")
                    input_dim=14
                    feature_dim=64 
                    instance_dim=8
                    cluster_dim=6

                trac_ik_right = TRACIK(base_link='base', tip_link='panda_1_hand_tcp', max_time=0.1)
                trac_ik_left = TRACIK(base_link='base', tip_link='panda_2_hand_tcp', max_time=0.1)

                start_pose_right = trac_ik_right.forward_kinematics(given_start_q[7:])
                start_pose_left = trac_ik_left.forward_kinematics(given_start_q[:7])
                goal_pose_right = trac_ik_right.forward_kinematics(given_goal_q[7:])
                goal_pose_left = trac_ik_left.forward_kinematics(given_goal_q[:7])

                start_ik_group = []
                goal_ik_group = []

                # IK start pose
                for _ in range(100):
                    joint_seed = np.random.uniform(constraint.lb[:7], constraint.ub[:7])
                    success_left, ik_left = trac_ik_left.solve(np.array(start_pose_left[0]), np.array(start_pose_left[1]),joint_seed)
                    if not success_left:
                        continue
                    if np.any(ik_left < constraint.lb[7:]) or np.any(ik_left > constraint.ub[7:]):
                        continue

                    success_right, ik_right = trac_ik_right.solve(np.array(start_pose_right[0]), np.array(start_pose_right[1]), joint_seed)
                    if not success_right:
                        continue
                    if np.any(ik_right < constraint.lb[:7]) or np.any(ik_right > constraint.ub[:7]):
                        continue
                    
                    start_full_ik = np.concatenate((ik_left, ik_right))
                    if constraint.planning_scene.is_valid(start_full_ik):
                        start_ik_group.append(start_full_ik)
                        # constraint.planning_scene.display(start_full_ik)

                # IK goal pose
                for _ in range(100):
                    joint_seed = np.random.uniform(constraint.lb[:7], constraint.ub[:7])
                    success_left, ik_left = trac_ik_left.solve(np.array(goal_pose_left[0]), np.array(goal_pose_left[1]),joint_seed)
                    if not success_left:
                        continue
                    if np.any(ik_left < constraint.lb[7:]) or np.any(ik_left > constraint.ub[7:]):
                        continue
                    success_right, ik_right = trac_ik_right.solve(np.array(goal_pose_right[0]), np.array(goal_pose_right[1]), joint_seed)
                    if not success_right:
                        continue
                    if np.any(ik_right < constraint.lb[:7]) or np.any(ik_right > constraint.ub[:7]):
                        continue
                    
                    goal_full_ik = np.concatenate((ik_left, ik_right))
                    if constraint.planning_scene.is_valid(goal_full_ik):
                        goal_ik_group.append(goal_full_ik)
                        # constraint.planning_scene.display(goal_full_ik)

            elif "panda_orientation" in exp_name:
                model_path = os.path.join('contrastiveik','save',"panda_orientation", "checkpoint_200.tar")
                input_dim=7
                feature_dim=32
                instance_dim=5
                cluster_dim=2

                trac_ik = TRACIK(base_link='panda_link0', tip_link='panda_hand_tcp', max_time=0.1)
                start_pose = trac_ik.forward_kinematics(given_start_q)
                goal_pose = trac_ik.forward_kinematics(given_goal_q)
                start_ik_group = []
                goal_ik_group = []

                for _ in range(100):
                    joint_seed = np.random.uniform(constraint.lb[:7], constraint.ub[:7])
                    success, ik = trac_ik.solve(np.array(start_pose[0]), np.array(start_pose[1]), joint_seed)
                    if not success:
                        continue
                    if np.any(ik < constraint.lb[:7]) or np.any(ik > constraint.ub[:7]):
                        continue
                    
                    if constraint.planning_scene.is_valid(ik):
                        start_ik_group.append(ik)
                        # constraint.planning_scene.display(ik)
                for _ in range(100):
                    joint_seed = np.random.uniform(constraint.lb[:7], constraint.ub[:7])
                    success, ik = trac_ik.solve(np.array(goal_pose[0]), np.array(goal_pose[1]), joint_seed)
                    if not success:
                        continue
                    if np.any(ik < constraint.lb[:7]) or np.any(ik > constraint.ub[:7]):
                        continue
                    
                    if constraint.planning_scene.is_valid(ik):
                        goal_ik_group.append(ik)
                        # constraint.planning_scene.display(ik)

            elif "panda_triple" in exp_name:
                
                # model_path = os.path.join('contrastiveik','save',"panda_triple_fixed_old", "checkpoint_1200.tar") # emd_test_1
                model_path = os.path.join('contrastiveik','save',"panda_triple_fixed", "checkpoint_7800.tar") # emd_test_0617
                input_dim=21 
                feature_dim=128 
                instance_dim=9
                cluster_dim=12

                trac_ik_left = TRACIK(base_link='base', tip_link='panda_left_hand_tcp', max_time=0.1)
                trac_ik_right = TRACIK(base_link='base', tip_link='panda_right_hand_tcp', max_time=0.1)
                trac_ik_top = TRACIK(base_link='base', tip_link='panda_top_hand_tcp', max_time=0.1)

                start_pose_left = trac_ik_left.forward_kinematics(given_start_q[:7])
                start_pose_right = trac_ik_right.forward_kinematics(given_start_q[7:14])
                start_pose_top = trac_ik_top.forward_kinematics(given_start_q[14:])
                
                goal_pose_left = trac_ik_left.forward_kinematics(given_goal_q[:7])
                goal_pose_right = trac_ik_right.forward_kinematics(given_goal_q[7:14])
                goal_pose_top = trac_ik_top.forward_kinematics(given_goal_q[14:])

                start_ik_group = []
                goal_ik_group = []

                # start_ik_group.append(given_start_q)
                # goal_ik_group.append(given_goal_q)
                
                # IK start pose
                for _ in range(100):
                    joint_seed = np.random.uniform(constraint.lb[:7], constraint.ub[:7])

                    success, ik_left = trac_ik_left.solve(np.array(start_pose_left[0]), np.array(start_pose_left[1]),joint_seed)
                    if not success:
                        continue
                    if np.any(ik_left < constraint.lb[:7]) or np.any(ik_left > constraint.ub[:7]):
                        continue

                    success, ik_right = trac_ik_right.solve(np.array(start_pose_right[0]), np.array(start_pose_right[1]), joint_seed)
                    if not success:
                        continue
                    if np.any(ik_right < constraint.lb[:7]) or np.any(ik_right > constraint.ub[:7]):
                        continue

                    success, ik_top = trac_ik_top.solve(np.array(start_pose_top[0]), np.array(start_pose_top[1]), joint_seed)
                    if not success:
                        continue
                    if np.any(ik_top < constraint.lb[:7]) or np.any(ik_top > constraint.ub[:7]):
                        continue

                    start_full_ik = np.concatenate((ik_left, ik_right, ik_top))
                    if constraint.planning_scene.is_valid(start_full_ik):
                        start_ik_group.append(start_full_ik)
                        # constraint.planning_scene.display(full_ik)
                
                # IK goal pose
                for _ in range(100):
                    joint_seed = np.random.uniform(constraint.lb[:7], constraint.ub[:7])
                    success, ik_left = trac_ik_left.solve(np.array(goal_pose_left[0]), np.array(goal_pose_left[1]),joint_seed)
                    if not success:
                        continue
                    if np.any(ik_left < constraint.lb[:7]) or np.any(ik_left > constraint.ub[:7]):
                        continue

                    success, ik_right = trac_ik_right.solve(np.array(goal_pose_right[0]), np.array(goal_pose_right[1]), joint_seed)
                    if not success:
                        continue
                    if np.any(ik_right < constraint.lb[:7]) or np.any(ik_right > constraint.ub[:7]):
                        continue

                    success, ik_top = trac_ik_top.solve(np.array(goal_pose_top[0]), np.array(goal_pose_top[1]), joint_seed)
                    if not success:
                        continue
                    if np.any(ik_top < constraint.lb[:7]) or np.any(ik_top > constraint.ub[:7]):
                        continue
                    
                    goal_full_ik = np.concatenate((ik_left, ik_right, ik_top))
                    if constraint.planning_scene.is_valid(goal_full_ik):
                        goal_ik_group.append(goal_full_ik)
                        # constraint.planning_scene.display(full_ik)
            
            elif "ur5_dual" in exp_name:
                if "ur5_dual_orientation" in exp_name:
                    model_path = os.path.join('contrastiveik','save',"ur5_dual_orientation_fixed", "checkpoint_10000.tar")
                    input_dim=12
                    feature_dim=64
                    instance_dim=8
                    cluster_dim=4
                else:
                    # model_path = os.path.join('contrastiveik','save',"ur5_dual_fixed", "checkpoint_1700.tar")
                    # model_path = os.path.join('contrastiveik','save',"ur5_dual_fixed_0725", "checkpoint_10000.tar")
                    model_path = os.path.join('contrastiveik','save',"ur5_dual_fixed_transformer", "checkpoint_10000.tar")
                    input_dim=12
                    feature_dim=64 
                    instance_dim=6
                    cluster_dim=6


                trac_ik_left = TRACIK(base_link='base', tip_link='R1_ur5_robotiq_85_gripper', max_time=0.1)
                trac_ik_right = TRACIK(base_link='base', tip_link='R2_ur5_robotiq_85_gripper', max_time=0.1)

                start_pose_right, start_pose_left = compute_dual_grasp_poses(obj_start_pose[:3], obj_start_pose[3:7], condition)
                goal_pose_right, goal_pose_left = compute_dual_grasp_poses(obj_goal_pose[:3], obj_goal_pose[3:7], condition)

                start_ik_group = []
                goal_ik_group = []

                # IK start pose
                for _ in range(100):
                    joint_seed = np.random.uniform(constraint.lb[:6], constraint.ub[:6])
                    success_left, ik_left = trac_ik_left.solve(np.array(start_pose_left[0]), np.array(start_pose_left[1]),joint_seed)
                    if not success_left:
                        continue
                    if np.any(ik_left < constraint.lb[6:]) or np.any(ik_left > constraint.ub[6:]):
                        continue
                    
                    success_right, ik_right = trac_ik_right.solve(np.array(start_pose_right[0]), np.array(start_pose_right[1]), joint_seed)
                    if not success_right:
                        continue
                    if np.any(ik_right < constraint.lb[:6]) or np.any(ik_right > constraint.ub[:6]):
                        continue
                    
                    start_full_ik = np.concatenate((ik_right, ik_left))
                    if constraint.planning_scene.is_valid(start_full_ik):
                        start_ik_group.append(start_full_ik)
                        # constraint.planning_scene.display(start_full_ik)
                
                # IK goal pose
                for _ in range(100):
                    joint_seed = np.random.uniform(constraint.lb[:6], constraint.ub[:6])
                    success_left, ik_left = trac_ik_left.solve(np.array(goal_pose_left[0]), np.array(goal_pose_left[1]),joint_seed)
                    if not success_left:
                        continue
                    if np.any(ik_left < constraint.lb[6:]) or np.any(ik_left > constraint.ub[6:]):
                        continue
                    
                    success_right, ik_right = trac_ik_right.solve(np.array(goal_pose_right[0]), np.array(goal_pose_right[1]), joint_seed)
                    if not success_right:
                        continue
                    if np.any(ik_right < constraint.lb[:6]) or np.any(ik_right > constraint.ub[:6]):
                        continue
                    
                    goal_full_ik = np.concatenate((ik_right,ik_left))
                    if constraint.planning_scene.is_valid(goal_full_ik):
                        goal_ik_group.append(goal_full_ik)
                        # constraint.planning_scene.display(goal_full_ik)
                
                # print("start_ik_group len: ", len(start_ik_group))
                # print("goal_ik_group len: ", len(goal_ik_group))
            else:
                raise NotImplementedError

            print("metric: ", args.metric)
            if args.metric == "min_q":
                minimum_q = np.inf
                for i in range(len(start_ik_group)):
                    for j in range(len(goal_ik_group)):
                        q_dist = np.linalg.norm(start_ik_group[i] - goal_ik_group[j])
                        if q_dist < minimum_q:
                            min_start_q = start_ik_group[i]
                            min_goal_q = goal_ik_group[j]
                            minimum_q = q_dist

                start_q = min_start_q
                goal_q = min_goal_q
                
            elif args.metric == "min_z" or args.metric == "min_embedding":
                if "transformer" in model_path:
                    model = network.TransformerNetwork(input_dim=input_dim, feature_dim=feature_dim, 
                               instance_dim=instance_dim, cluster_dim=cluster_dim)
                else:
                    model = network.SimNetwork(input_dim=input_dim, feature_dim=feature_dim, 
                               instance_dim=instance_dim, cluster_dim=cluster_dim)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.load_state_dict(torch.load(model_path, map_location=device.type)['net'])
                model.to(device)
                model.eval()
                if args.metric == "min_z":
                    with torch.no_grad():
                        z_start_ik = model.feature_inference(torch.tensor(np.array(start_ik_group), dtype=torch.float32).to(device)).cpu().numpy()
                        z_goal_ik = model.feature_inference(torch.tensor(np.array(goal_ik_group), dtype=torch.float32).to(device)).cpu().numpy()
                else:
                    with torch.no_grad():
                        z_start_ik = model.inference(torch.tensor(np.array(start_ik_group), dtype=torch.float32).to(device)).cpu().numpy()
                        z_goal_ik = model.inference(torch.tensor(np.array(goal_ik_group), dtype=torch.float32).to(device)).cpu().numpy()

                minimum_z = np.inf
                for i in range(len(z_start_ik)):
                    for j in range(len(z_goal_ik)):
                        z_dist = np.linalg.norm(z_start_ik[i] - z_goal_ik[j])
                        if z_dist < minimum_z:
                            min_start_q = start_ik_group[i]
                            min_goal_q = goal_ik_group[j]
                            minimum_z = z_dist
                
                start_q = min_start_q
                goal_q = min_goal_q

            elif args.metric == "min_q_z" or args.metric == "min_q_embedding":
                if "transformer" in model_path:
                    model = network.TransformerNetwork(input_dim=input_dim, feature_dim=feature_dim, 
                               instance_dim=instance_dim, cluster_dim=cluster_dim)
                else:
                    model = network.SimNetwork(input_dim=input_dim, feature_dim=feature_dim, 
                               instance_dim=instance_dim, cluster_dim=cluster_dim)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.load_state_dict(torch.load(model_path, map_location=device.type)['net'])
                model.to(device)
                model.eval()
                if args.metric == "min_q_z":
                    with torch.no_grad():
                            z_start_ik = model.feature_inference(torch.tensor(np.array(start_ik_group), dtype=torch.float32).to(device)).cpu().numpy()
                            z_goal_ik = model.feature_inference(torch.tensor(np.array(goal_ik_group), dtype=torch.float32).to(device)).cpu().numpy()
                else:
                    with torch.no_grad():
                        z_start_ik = model.inference(torch.tensor(np.array(start_ik_group), dtype=torch.float32).to(device)).cpu().numpy()
                        z_goal_ik = model.inference(torch.tensor(np.array(goal_ik_group), dtype=torch.float32).to(device)).cpu().numpy()
                
                min_total_dist = np.inf
                for i in range(len(start_ik_group)):
                    for j in range(len(goal_ik_group)):
                        q_dist = np.linalg.norm(start_ik_group[i] - goal_ik_group[j])
                        z_dist = np.linalg.norm(z_start_ik[i] - z_goal_ik[j])
                        total_dist = q_dist + z_dist  # 또는 alpha * q_dist + beta * z_dist

                        if total_dist < min_total_dist:
                            min_start_q = start_ik_group[i]
                            min_goal_q = goal_ik_group[j]
                            min_total_dist = total_dist

                start_q = min_start_q
                goal_q = min_goal_q

            elif args.metric == "random":
                start_q = start_ik_group[np.random.randint(0, len(start_ik_group))]
                goal_q = goal_ik_group[np.random.randint(0, len(goal_ik_group))]
            
        else:
            start_q = given_start_q
            goal_q = given_goal_q
        ####################
        # # joint_dists = []
        # # latent_dists = []
        # # highlight_point = None

        # # for i in range(len(start_ik_group)):
        # #     for j in range(len(goal_ik_group)):
        # #         q_dist = np.linalg.norm(start_ik_group[i] - goal_ik_group[j])
        # #         z_dist = np.linalg.norm(z_start_ik[i] - z_goal_ik[j])
                
        # #         joint_dists.append(q_dist)
        # #         latent_dists.append(z_dist)

        # #         # 현재 쌍이 start_q, goal_q이면 강조할 포인트로 저장
        # #         if np.allclose(start_ik_group[i], start_q) and np.allclose(goal_ik_group[j], goal_q):
        # #             highlight_point = (q_dist, z_dist)

        # # Plot start IK group joint values: joint i vs joint j for all pairs (i, j)
        # import matplotlib.pyplot as plt

        # # Plot joint space pairwise scatter and diagonal histograms for start_ik_group
        # # start_ik_group = np.array(start_ik_group)
        # # n_joints = start_ik_group.shape[1]

        # # fig, axes = plt.subplots(n_joints, n_joints, figsize=(2 * n_joints, 2 * n_joints))
        # # fig.suptitle('Start IK Group: Joint Pairwise Scatter & Diagonal Histograms', fontsize=14)

        # # for i in range(n_joints):
        # #     for j in range(n_joints):
        # #         ax = axes[i, j]
        # #         if i == j:
        # #             ax.hist(start_ik_group[:, i], bins=20, color='skyblue', alpha=0.7)
        # #         else:
        # #             ax.scatter(start_ik_group[:, j], start_ik_group[:, i], alpha=0.3, s=5)
        # #         if i < n_joints - 1:
        # #             ax.set_xticklabels([])
        # #         else:
        # #             ax.set_xlabel(f'Joint {j}', fontsize=8)
        # #         if j > 0:
        # #             ax.set_yticklabels([])
        # #         else:
        # #             ax.set_ylabel(f'Joint {i}', fontsize=8)
        # #         ax.tick_params(axis='both', which='major', labelsize=6)
        # # plt.tight_layout(rect=[0, 0, 1, 0.96])
        # # plt.show()

        # # Plot latent space pairwise scatter and diagonal histograms for z_start_ik
        # print("c")
        # n_latent = z_start_ik.shape[1]
        # print("n_latent: ", n_latent)
        # print("b")  
        # fig, axes = plt.subplots(n_latent, n_latent, figsize=(2 * n_latent, 2 * n_latent))
        # fig.suptitle('z_start_ik: Latent Pairwise Scatter & Diagonal Histograms', fontsize=14)
        # print("a")
        # for i in range(n_latent):
        #     for j in range(n_latent):
        #         ax = axes[i, j]
        #         if i == j:
        #             ax.hist(z_start_ik[:, i], bins=20, color='lightcoral', alpha=0.7)
        #         else:
        #             ax.scatter(z_start_ik[:, j], z_start_ik[:, i], alpha=0.3, s=5)
        #         if i < n_latent - 1:
        #             ax.set_xticklabels([])
        #         else:
        #             ax.set_xlabel(f'Latent {j}', fontsize=8)
        #         if j > 0:
        #             ax.set_yticklabels([])
        #         else:
        #             ax.set_ylabel(f'Latent {i}', fontsize=8)
        #         ax.tick_params(axis='both', which='major', labelsize=6)
        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.show()
        
        # # Plot
        # plt.figure(figsize=(8, 6))
        # plt.scatter(joint_dists, latent_dists, alpha=0.5, label='All IK pairs')
        # if highlight_point:
        #     plt.scatter(*highlight_point, color='red', s=100, edgecolors='black', label='Chosen pair (start_q, goal_q)')

        # plt.xlabel('Joint Space Distance ||q_start - q_goal||')
        # plt.ylabel('Latent Space Distance ||z_start - z_goal||')
        # plt.title('Joint vs Latent Distance between IK Pairs')
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        # plt.figure(figsize=(8, 6))
        # plt.scatter(range(len(z_start_ik)), z_start_ik[:, 0], label='z_start_ik dim 0')
        # if z_start_ik.shape[1] > 1:
        #     for dim in range(1, z_start_ik.shape[1]):
        #         plt.scatter(range(len(z_start_ik)), z_start_ik[:, dim], label=f'z_start_ik dim {dim}')
        # plt.xlabel('IK Sample Index')
        # plt.ylabel('Latent Value')
        # plt.title('Latent Space Values of z_start_ik')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # continue
        ####################
        latent_jump = False
        if 'latent_jump' in method:
            latent_jump = True

        for trial in range(trials):
            if 'latent_rrt' in method:

                if use_given_start_goal:
                    planner = ConstrainedLatentBiRRT(constraint_model, validity_model, constraint, latent_jump=latent_jump)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    lrs_start = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_start.set_target_pose(start_pose)
                    lrs_goal = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_goal.set_target_pose(goal_pose)

                    planner = ConstrainedLatentBiRRT(constraint_model, validity_model, constraint, latent_jump=latent_jump,
                                                    start_region_fn=lrs_start.sample, 
                                                    goal_region_fn=lrs_goal.sample)
                planner.max_distance = model_info['planning']['max_distance_q'] / model_info['planning']['alpha']
                planner.max_distance_q = model_info['planning']['max_distance_q']
                planner.off_manifold_threshold = model_info['planning']['off_manifold_threshold']
                planner.p_q_plan = model_info['planning']['p_q_plan']
                planner.debug = debug
                r, z_path, q_path, path_ref = planner.solve(max_time=max_time)
            
            elif method == 'sampling_rrt':
                if use_given_start_goal:
                    planner = SampleBiasedConstrainedBiRRT(state_dim=model_info['x_dim'], model=constraint_model, constraint=constraint)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    lrs_start = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_start.set_target_pose(start_pose)
                    lrs_goal = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_goal.set_target_pose(goal_pose)
                    planner = SampleBiasedConstrainedBiRRT(state_dim=model_info['x_dim'], model=constraint_model, constraint=constraint,
                                                        start_region_fn=lrs_start.sample, 
                                                        goal_region_fn=lrs_goal.sample)
                planner.max_distance = model_info['planning']['max_distance_q']
                planner.qnew_threshold = model_info['planning']['qnew_threshold']
                planner.debug = debug
                r, q_path = planner.solve(max_time=max_time)

            elif method == 'constrained_rrt':
                if use_given_start_goal:
                    planner = ConstrainedBiRRT(state_dim=model_info['x_dim'], constraint=constraint)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    rs_start = RegionSampler(constraint)
                    rs_start.set_target_pose(start_pose)
                    rs_goal = RegionSampler(constraint)
                    rs_goal.set_target_pose(goal_pose)
                    planner = ConstrainedBiRRT(state_dim=model_info['x_dim'],
                                               constraint=constraint,
                                               start_region_fn=rs_start.sample,
                                               goal_region_fn=rs_goal.sample)

                planner.max_distance = model_info['planning']['max_distance_q']
                planner.debug = debug
                r, q_path = planner.solve(max_time=max_time)

            elif method == 'precomputed_roadmap_prm':
                if use_given_start_goal:
                    planner = PrecomputedRoadmap(state_dim=model_info['x_dim'], constraint=constraint)
                    planner.set_graph(graph=precomputed_roadmap)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    raise NotImplementedError
                
                planner.debug = debug
                planner.max_distance = model_info['planning']['max_distance_q']
                r, q_path = planner.solve(max_time=max_time)

            elif method == 'precomputed_graph_rrt':
                if use_given_start_goal:
                    planner = PrecomputedGraph(state_dim=model_info['x_dim'], constraint=constraint)
                    planner.set_graph(graph=precomputed_graph)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    raise NotImplementedError
                
                planner.debug = debug
                planner.max_distance = model_info['planning']['max_distance_q']
                r, q_path = planner.solve(max_time=max_time)
            

            else:
                raise NotImplementedError

            if debug:
                print('planning time', planner.solved_time)

            test_cnt += 1
            
            if r is True:
                path_length = np.array([np.linalg.norm(q_path[i+1] - q_path[i]) for i in range(len(q_path)-1)]).sum()
                test_suc_cnt += 1
                solved_time = planner.solved_time
            else:
                if debug:
                    print('failed to find a path')
                q_path = None
                z_path = None
                path_ref = None

                solved_time = -1.0
                path_length = -1.0

            test_paths.append(q_path)
            test_times.append(solved_time)
            test_path_lenghts.append(path_length)

            if 'latent_rrt' in method:
                test_paths_z.append(z_path)
                test_path_refs.append(path_ref)

            mean_test_times = np.mean(test_times, where=np.array(test_times) > 0)
            mean_test_path_lenghts = np.mean(test_path_lenghts, where=np.array(test_path_lenghts) > 0)
            
            tq.set_description('test suc rate: {:.3f}, avg time: {:.3f}, avg path length: {:.3f}'.format(test_suc_cnt/test_cnt, mean_test_times, mean_test_path_lenghts))

            if r is True:
                if display:
                    hz = 40
                    duration, qs_sample, qds_sample, qdds_sample, ts_sample = time_parameterize(q_path, model_info, hz=hz)
                    
                    if debug:
                        print('duration', duration)

                    for q in qs_sample:
                        constraint.planning_scene.display(q)
                        time.sleep(2.0/hz)
                
    test_paths_cartesian = []
    for path in test_paths:
        if path is None:
            test_paths_cartesian.append(None)
            continue
        
        path_cartesian = []
        for q in path:
            cur_idx = 0
            cartesian_vector = []
            for arm_name, dof in zip(constraint.arm_names, constraint.arm_dofs):
                pos, quat = constraint.forward_kinematics(arm_name, q[cur_idx:cur_idx+dof])
                cur_idx += dof
                cartesian_vector.append(np.concatenate([pos, quat]))
                
            cartesian_vector = np.concatenate(cartesian_vector)
            path_cartesian.append(cartesian_vector)
        test_paths_cartesian.append(path_cartesian)


    mean_test_times = np.mean(test_times, where=np.array(test_times) > 0)
    std_test_times = np.std(test_times, where=np.array(test_times) > 0)
    mean_test_path_lenghts = np.mean(test_path_lenghts, where=np.array(test_path_lenghts) > 0)
    std_test_path_lenghts = np.std(test_path_lenghts, where=np.array(test_path_lenghts) > 0)

    ret =  {'experiment_name': exp_name,
            'model_tag_name': model_info['constraint_model']['tag'],
            'method': method,
            'use_given_start_goal': use_given_start_goal,
            'max_time': max_time,

            # test scene info
            'test_scene_start_idx': test_scene_start_idx,
            'test_scene_cnt': num_test_scenes,
            
            # result overview
            'test_cnt': test_cnt, 
            'test_suc_cnt': test_suc_cnt, 
            'success_rate': test_suc_cnt/test_cnt,
            'mean_test_times': mean_test_times,
            'std_test_times': std_test_times,
            'mean_test_path_lenghts': mean_test_path_lenghts,
            'std_test_path_lenghts': std_test_path_lenghts,

            # result details
            'test_times': test_times, 
            'test_paths': test_paths, 
            'test_path_lenghts': test_path_lenghts, 
            'test_paths_cartesian': test_paths_cartesian}
    
    if 'latent_rrt' in method:
        ret['test_paths_z'] = test_paths_z
        ret['test_path_refs'] = test_path_refs

    return ret

def benchmark_ik_selection_all_pairs(
    model_info,
    constraint_model,
    validity_model,
    constraint,
    test_scenes,
    model_path,
    method="latent_rrt",
    use_given_start_goal=False,
    max_time=30.0,
    debug=False,
):
    """
    For each test scene, run the planner for all possible combinations of start and goal IK pairs.
    Returns a dict with per-scene results for all (start_q, goal_q) pairs.
    """
    import numpy as np
    import torch
    from ljcmp.utils import network

    results = {}
    for scene_idx, scene in enumerate(test_scenes):
        scene_result = {}
        # Extract poses and/or given qs
        start_pose = scene.get("start_pose")
        goal_pose = scene.get("goal_pose")
        given_start_q = scene.get("given_start_q", None)
        given_goal_q = scene.get("given_goal_q", None)

        # Generate all IK solutions for start and goal
        trac_ik_left = constraint.trac_ik_left
        trac_ik_right = constraint.trac_ik_right
        constraint_lb = constraint.lb
        constraint_ub = constraint.ub

        # Helper to get IK group for a pose
        def get_ik_group(pose, trac_ik_left, trac_ik_right, constraint, n_samples=100):
            ik_group = []
            for _ in range(n_samples):
                joint_seed = np.random.uniform(constraint.lb[:6], constraint.ub[:6])
                success_left, ik_left = trac_ik_left.solve(np.array(pose[0]), np.array(pose[1]), joint_seed)
                if not success_left:
                    continue
                if np.any(ik_left < constraint.lb[6:]) or np.any(ik_left > constraint.ub[6:]):
                    continue
                success_right, ik_right = trac_ik_right.solve(np.array(pose[0]), np.array(pose[1]), joint_seed)
                if not success_right:
                    continue
                if np.any(ik_right < constraint.lb[:6]) or np.any(ik_right > constraint.ub[:6]):
                    continue
                full_ik = np.concatenate((ik_right, ik_left))
                if constraint.planning_scene.is_valid(full_ik):
                    ik_group.append(full_ik)
            return ik_group

        if not use_given_start_goal:
            start_ik_group = get_ik_group(start_pose, trac_ik_left, trac_ik_right, constraint)
            goal_ik_group = get_ik_group(goal_pose, trac_ik_left, trac_ik_right, constraint)
        else:
            # If using given qs, just use those as the only "group"
            start_ik_group = [given_start_q]
            goal_ik_group = [given_goal_q]

        # For all combinations of start and goal IKs, run the planner
        pair_results = {}
        for i, start_q in enumerate(start_ik_group):
            for j, goal_q in enumerate(goal_ik_group):
                ret = benchmark(
                    model_info=model_info,
                    constraint_model=constraint_model,
                    validity_model=validity_model,
                    constraint=constraint,
                    exp_name=model_info.get("exp_name", "exp"),
                    model_path=model_path,
                    method=method,
                    use_given_start_goal=True,  # always use the explicit pair
                    max_time=max_time,
                    test_scene_start_idx=scene_idx,
                    num_test_scenes=1,
                    debug=debug,
                    metric="given",  # not using metric, explicit pair
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    given_start_q=start_q,
                    given_goal_q=goal_q,
                )
                pair_results[(i, j)] = ret
        scene_result["ik_pair_results"] = pair_results
        scene_result["start_ik_group"] = start_ik_group
        scene_result["goal_ik_group"] = goal_ik_group
        results[scene_idx] = scene_result
    return results