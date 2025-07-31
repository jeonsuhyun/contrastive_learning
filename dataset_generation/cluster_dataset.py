import numpy as np

import pickle, os
import pandas as pd
from sklearn.cluster import DBSCAN
import tqdm
from scipy.linalg import null_space
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='panda_triple', help='panda_orientation, panda_dual, panda_dual_orientation')
parser.add_argument('--data_file_name', '-D', type=str, default='data_fixed_100000.npy')
parser.add_argument('--epsilon', type=float, default=1.8)
parser.add_argument('--min_samples', type=int, default=10)
parser.add_argument('--max_size', type=int, default=10000)
parser.add_argument('--surfix', '-S', type=str, default='')

args = parser.parse_args()

data_file_name = os.path.join('dataset', args.exp_name, 'manifold', args.data_file_name)
null_file_name = os.path.join('dataset', args.exp_name, 'manifold', args.data_file_name.replace('data', 'null'))

"""
DBSCAN clustering
"""

dataset = np.load(data_file_name)
joint = dataset[:,3:]
null = np.load(null_file_name)

epsilon = args.epsilon
while True:
    print('epsilon: ', epsilon)
    model = DBSCAN(min_samples=args.min_samples, eps=epsilon).fit(joint)
    print('labels\n', model.labels_)
    print(max(model.labels_))
    print(min(model.labels_))
    label_nums = [len(model.labels_[model.labels_==i]) for i in range(max(model.labels_))]
    print('label_nums\n', label_nums)
    import pdb; pdb.set_trace()
    if max(label_nums) > args.max_size:
        break

    print('size is too small, increasing epsilon')
    print('max size: ', max(label_nums))
    print('max label num: ', max(label_nums))
    epsilon += 0.1

print('final epsilon: ', epsilon)
print(label_nums)
print('max: ', np.argmax(label_nums))

# print('und', model.labels_[model.labels_==-1])

model_path = os.path.join('model', args.exp_name, 'dbscan')
os.makedirs(model_path, exist_ok=True)

with open(os.path.join(model_path, 'dbscan.pkl'), 'wb') as f:
    pickle.dump(model, f)

import pdb; pdb.set_trace()

# Save all clusters with their labels
all_clusters_path = os.path.join('dataset', args.exp_name, 'manifold', f'all_clusters{args.surfix}.pkl')
all_clusters = {label: joint[model.labels_ == label] for label in set(model.labels_) if label != -1}

with open(all_clusters_path, 'wb') as f:
    pickle.dump(all_clusters, f)



# save largest cluster
largest_cluster = joint[model.labels_==np.argmax(label_nums)]
data_largest = joint[model.labels_==np.argmax(label_nums)]
null_largest = null[model.labels_==np.argmax(label_nums)]

largest_data_file_name = os.path.join('dataset', args.exp_name, 'manifold', f'data_{args.max_size}{args.surfix}_clustered.npy')
largest_null_file_name = os.path.join('dataset', args.exp_name, 'manifold', f'null_{args.max_size}{args.surfix}_clustered.npy')

np.save(largest_data_file_name, data_largest[:args.max_size])
np.save(largest_null_file_name, null_largest[:args.max_size]) 