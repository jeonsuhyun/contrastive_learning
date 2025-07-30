import os
import yaml
import numpy as np
import torch
import pickle

import argparse

from termcolor import colored

from ljcmp.utils.model_utils import benchmark
from ljcmp.utils.generate_environment import generate_environment

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-E', type=str, default='ur5_dual', help='panda_orientation, panda_dual, panda_dual_orientation, panda_triple')
parser.add_argument('--seed', type=int, default=1107)
parser.add_argument('--use_given_start_goal', type=bool, default=True)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--display', type=bool, default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--method', '-M', type=str, default='constrained_rrt', help='latent_rrt, latent_rrt_latent_jump, sampling_rrt, precomputed_roadmap_prm, precomputed_graph_rrt, project_rrt')
parser.add_argument('--test_scene_start_idx', type=int, default=500)
parser.add_argument('--metric', type=str, default='min_q', help='min_q, min_z')
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--num_test_scenes', type=int, default=100)
parser.add_argument('--max_time', type=float, default=100.0)
parser.add_argument('--trials', type=int, default=1)

args = parser.parse_args()
if args.device == 'cuda':
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
        print(colored('CUDA is not available, use CPU instead', 'red'))
print(colored('Using device: {}'.format(args.device), 'green'))

device = args.device
np.random.seed(args.seed)
torch.manual_seed(args.seed)

constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(args.exp_name)

print(colored(' ---- Start benchmarking ----', 'green'))
print('exp_name :', args.exp_name)
print('tag      :', model_info['constraint_model']['tag'])
print('method   :', args.method)

np.set_printoptions(precision=6, suppress=True)

results = benchmark(args = args,
                    exp_name=args.exp_name,
                    model_info=model_info,
                    method=args.method,
                    constraint=constraint,
                    device=device,
                    condition=condition,
                    max_time = args.max_time,
                    use_given_start_goal=args.use_given_start_goal,
                    update_scene_from_yaml=update_scene_from_yaml,
                    debug=args.debug,
                    display=args.display,
                    trials=args.trials,
                    test_scene_start_idx=args.test_scene_start_idx,
                    num_test_scenes=args.num_test_scenes,
                    load_validity_model=True)

model_tag = model_info['constraint_model']['tag']
result_save_dir = f'result/emd_test_0617/{args.exp_name}/{args.method}/{args.metric}/'
os.makedirs(result_save_dir, exist_ok=True)

print(colored(' ---- Benchmarking finished ----', 'green'))
print('test suc rate', results['success_rate'])
print('avg time', results['mean_test_times'])
print('std time', results['std_test_times'])
print('avg path length', results['mean_test_path_lenghts'])
print('std path length', results['std_test_path_lenghts'])

print(colored(' ---- Saving results ----', 'green'))
print('result_save_dir', result_save_dir)

pickle.dump(results, open(f'{result_save_dir}/test_result.pkl', 'wb'))

results_overview = {'success_rate': results['success_rate'],
                    'mean_test_times': results['mean_test_times'].tolist(),
                    'std_test_times': results['std_test_times'].tolist(),
                    'mean_test_path_lenghts': results['mean_test_path_lenghts'].tolist(),
                    'std_test_path_lenghts': results['std_test_path_lenghts'].tolist()}

yaml.dump(results_overview, open(f'{result_save_dir}/test_result_overview.yaml', 'w'))
