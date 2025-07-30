import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os

# Example data

# Load data from /result/emd_test file
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-E', type=str, default='panda_dual', 
                    help='panda_orientation,'
                    'panda_dual,'
                    'panda_dual_orientation,'
                    'panda_triple')

args = parser.parse_args()
dir_path = os.path.join('./result/emd_test_1',args.exp_name,'latent_rrt')

val_test_time = []
val_test_length = []

scenario_names = ['min_z', 'min_q', 'random', 'given']
for scenario in scenario_names:
    file_path = f'{dir_path}/{scenario}/test_result.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        val_test_time.append(data['test_times'])
        val_test_length.append(data['test_path_lenghts'])

valid_indices_0 = [i for i in range(len(val_test_time[0])) if val_test_time[0][i] != -1]
valid_indices_1 = [i for i in range(len(val_test_time[1])) if val_test_time[1][i] != -1]
valid_indices = [i for i in range(len(val_test_time[0])) if val_test_time[0][i] != -1 and val_test_time[1][i] != -1]

print("Valid indices:", len(valid_indices))
print("Valid indices 0:", len(valid_indices_0))
print("Valid indices 1:", len(valid_indices_1))

valid_time_0 = [val_test_time[0][i] for i in valid_indices]
valid_time_1 = [val_test_time[1][i] for i in valid_indices]

valid_length_0 = [val_test_length[0][i] for i in valid_indices]
valid_length_1 = [val_test_length[1][i] for i in valid_indices]

print("valid time avg", np.mean(valid_time_0), np.mean(valid_time_1))
print("valid length avg", np.mean(valid_length_0), np.mean(valid_length_1))

test_time = []
test_length = []
avg_time = []
avg_length = []

for scenario in scenario_names:
    file_path = f'{dir_path}/{scenario}/test_result.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
        test_times_filtered = [t for t in data['test_times'] if t != -1]
        test_lengths_filtered = [l for l in data['test_path_lenghts'] if l != -1]
        test_time.append(test_times_filtered)
        test_length.append(test_lengths_filtered)
        
        avg_time.append(data['mean_test_times'])
        avg_length.append(data['mean_test_path_lenghts'])

print("Average Time:", avg_time)
print("Average Length:", avg_length)

# Plot time data as box plots
plt.figure(figsize=(10, 5))
plt.boxplot(test_time, labels=scenario_names, patch_artist=True)
plt.title('Time Data Distribution')
plt.xlabel('Scenarios')
plt.ylabel('Time (s)')
plt.ylim(0, 20)  # Set y-axis limit to 100
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('time_boxplot.png')  # Save the plot as an image
plt.show()

# Plot path length data as box plots
plt.figure(figsize=(10, 5))
plt.boxplot(test_length, labels=scenario_names, patch_artist=True, boxprops=dict(facecolor='orange'))
plt.title('Path Length Data Distribution')
plt.xlabel('Scenarios')
plt.ylabel('Path Length (units)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('path_length_boxplot.png')  # Save the plot as an image
plt.show()
