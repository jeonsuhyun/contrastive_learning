

import numpy as np
import torch
import yaml
from ljcmp.planning.distance_functions import distance_q, distance_z
from ljcmp.utils.model_utils import load_model

num_pairs = 10000
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)

exp_name = 'panda_orientation'
c = np.array([])

# exp_name = 'panda_dual'
# c = np.array([0.3, 0.05, 0.9])

# exp_name = 'panda_dual_orientation'
# c = np.array([0.3, 0.05, 0.9])

# exp_name = 'panda_triple'
# c = np.array([0.3, 0.6, 0.3])

model_info = yaml.load(open(f'model/{exp_name}/model_info.yaml', 'r'), Loader=yaml.FullLoader)

constraint_model, validity_model = load_model(exp_name, model_info, True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

constraint_model = constraint_model.to(device)
validity_model = validity_model.to(device)

z = torch.randn(num_pairs*2, model_info['z_dim']).to(device)
constraint_model.set_condition(c)
q = constraint_model.decode(z).detach().cpu().numpy()

z = z.detach().cpu().numpy()

distances_z = []
distances_z_no_cdf = []
distances_q = []
for i in range(num_pairs):
    distances_q.append(distance_q(q[i], q[i+1]))
    distances_z_no_cdf.append(distance_q(z[i], z[i+1]))
    distances_z.append(distance_z(z[i], z[i+1]))

distances_q = np.array(distances_q)
distances_z = np.array(distances_z)
distances_z_no_cdf = np.array(distances_z_no_cdf)   

alpha = np.mean(distances_q)/np.mean(distances_z)
alpha_non_cdf = np.mean(distances_q)/np.mean(distances_z_no_cdf)

normalized_distances_z_in_q = alpha*distances_z / distances_q
normalized_distances_z_in_q_non_cdf = alpha_non_cdf*distances_z_no_cdf / distances_q

print('normalized_distances_z_in_q', normalized_distances_z_in_q)
print('distances_q', distances_q)
print('distances_z', distances_z)

print('Mean distance in z space: {}'.format(np.mean(distances_z)))
print('Mean distance in q space: {}'.format(np.mean(distances_q)))

print('dist_zq std dev: {}'.format(np.std(normalized_distances_z_in_q)))
print('dist_zq std dev non cdf: {}'.format(np.std(normalized_distances_z_in_q_non_cdf)))

print('alpha: {}'.format(alpha))
print('alpha non cdf: {}'.format(alpha_non_cdf))    
