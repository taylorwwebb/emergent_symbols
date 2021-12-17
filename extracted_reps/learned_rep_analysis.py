import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Rectangle

# Settings
parser = argparse.ArgumentParser()
# Run number
parser.add_argument('--run', type=str, default='1')
args = parser.parse_args()

# Load training data
train_fname = './run' + args.run +'_train_reps.npz'
train_data = np.load(train_fname)
train_M_k = train_data['M_k']
train_key_r = train_data['all_key_r']
train_seq_ind = train_data['seq']

# Load test data
test_fname = './run' + args.run +'_test_reps.npz'
test_data = np.load(test_fname)
test_M_k = test_data['M_k']
test_key_r = test_data['all_key_r']
test_seq_ind = test_data['seq']

# Limit memory to first 9 time steps
train_M_k = train_M_k[:,:9,:]
test_M_k = test_M_k[:,:9,:]

# Stack all memory vectors 
train_M_k_stacked = []
for t in range(train_M_k.shape[1]):
	train_M_k_stacked.append(train_M_k[:,t,:])
train_M_k_stacked = np.concatenate(train_M_k_stacked, 0)
test_M_k_stacked = []
for t in range(test_M_k.shape[1]):
	test_M_k_stacked.append(test_M_k[:,t,:])
test_M_k_stacked = np.concatenate(test_M_k_stacked, 0)
all_M_k_stacked = np.concatenate([train_M_k_stacked, test_M_k_stacked], 0)

# Perform common PCA
pca = PCA()
pca.fit(all_M_k_stacked)

# Colors for plot
all_colors = ['red', 'teal', 'darkblue', 'darkviolet', 'orange', 'limegreen', 'fuchsia', 'yellow', 'darkgrey']
# Marker size
marker_size = 10

# Plot key_w for training set, t=1-9
legend_text = []
ax = plt.subplot(111)
for t in range(9):
	train_mem_t_PCA = pca.transform(train_M_k[:,t,:])
	plt.scatter(train_mem_t_PCA[:,0], train_mem_t_PCA[:,1], s=marker_size, color=all_colors[t])
	legend_text.append('t = ' + str(t+1))
plt.xlabel('1st PC')
plt.ylabel('2nd PC')
plt.legend(legend_text)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plot_fname = './key_w_t1to9_train.png'
plt.savefig(plot_fname)
plt.close()

# Colors for plot
all_colors = ['darkviolet', 'orange', 'limegreen', 'red', 'teal', 'darkblue']

# Plot key_w for training and test sets overlaid, t=1-3
all_items = []
ax = plt.subplot(111)
for t in range(3):
	test_mem_t_PCA = pca.transform(test_M_k[:,t,:])
	item = ax.scatter(test_mem_t_PCA[:,0], test_mem_t_PCA[:,1], s=marker_size, color=all_colors[t])
	all_items.append(item)
for t in range(3):
	train_mem_t_PCA = pca.transform(train_M_k[:,t,:])
	item = ax.scatter(train_mem_t_PCA[:,0], train_mem_t_PCA[:,1], s=marker_size, color=all_colors[t+3])
	all_items.append(item)
plt.xlabel('1st PC')
plt.ylabel('2nd PC')
key_w_xlim = ax.get_xlim()
key_w_ylim = ax.get_ylim()
extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
legend_handle = [extra, extra, extra, extra, extra, all_items[3], all_items[4], all_items[5], extra, all_items[0], all_items[1], all_items[2]]
legend_labels = ['', 't = 1', 't = 2', 't = 3', 'training', '', '', '', 'test', '', '', '']
ax.legend(legend_handle, legend_labels, ncol = 3, handletextpad = -2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plot_fname = './key_w_t123_train_vs_test.png'
plt.savefig(plot_fname)
plt.close()

# Compare M_k to retrieved memories (training set)
train_first_match_ind = np.argmax((np.expand_dims(train_seq_ind[:,0],1) == train_seq_ind[:,3:]).astype(np.float),1) + 2
train_second_match_ind = np.argmax((np.expand_dims(train_seq_ind[:,1],1) == train_seq_ind[:,3:]).astype(np.float),1) + 2
train_third_match_ind = np.argmax((np.expand_dims(train_seq_ind[:,2],1) == train_seq_ind[:,3:]).astype(np.float),1) + 2
train_key_r_first_match = []
train_key_r_second_match = []
train_key_r_third_match = []
for n in range(train_key_r.shape[0]):
	train_key_r_first_match.append(train_key_r[n,train_first_match_ind[n],:])
	train_key_r_second_match.append(train_key_r[n,train_second_match_ind[n],:])
	train_key_r_third_match.append(train_key_r[n,train_third_match_ind[n],:])
train_key_r_first_match = np.array(train_key_r_first_match)
train_key_r_second_match = np.array(train_key_r_second_match)
train_key_r_third_match = np.array(train_key_r_third_match)
# Discard confidence values
train_key_r_first_match = train_key_r_first_match[:,:-1]
train_key_r_second_match = train_key_r_second_match[:,:-1]
train_key_r_third_match = train_key_r_third_match[:,:-1]

# Compare M_k to retrieved memories (test set)
test_first_match_ind = np.argmax((np.expand_dims(test_seq_ind[:,0],1) == test_seq_ind[:,3:]).astype(np.float),1) + 2
test_second_match_ind = np.argmax((np.expand_dims(test_seq_ind[:,1],1) == test_seq_ind[:,3:]).astype(np.float),1) + 2
test_third_match_ind = np.argmax((np.expand_dims(test_seq_ind[:,2],1) == test_seq_ind[:,3:]).astype(np.float),1) + 2
test_key_r_first_match = []
test_key_r_second_match = []
test_key_r_third_match = []
for n in range(test_key_r.shape[0]):
	test_key_r_first_match.append(test_key_r[n,test_first_match_ind[n],:])
	test_key_r_second_match.append(test_key_r[n,test_second_match_ind[n],:])
	test_key_r_third_match.append(test_key_r[n,test_third_match_ind[n],:])
test_key_r_first_match = np.array(test_key_r_first_match)
test_key_r_second_match = np.array(test_key_r_second_match)
test_key_r_third_match = np.array(test_key_r_third_match)
# Discard confidence values
test_key_r_first_match = test_key_r_first_match[:,:-1]
test_key_r_second_match = test_key_r_second_match[:,:-1]
test_key_r_third_match = test_key_r_third_match[:,:-1]

# Plot retrieved memories for training vs. test
ax = plt.subplot(111)
test_key_r_first_PCA = pca.transform(test_key_r_first_match)
plt.scatter(test_key_r_first_PCA[:,0], test_key_r_first_PCA[:,1], s=marker_size, color=all_colors[0])
test_key_r_second_PCA = pca.transform(test_key_r_second_match)
plt.scatter(test_key_r_second_PCA[:,0], test_key_r_second_PCA[:,1], s=marker_size, color=all_colors[1])
test_key_r_third_PCA = pca.transform(test_key_r_third_match)
plt.scatter(test_key_r_third_PCA[:,0], test_key_r_third_PCA[:,1], s=marker_size, color=all_colors[2])
train_key_r_first_PCA = pca.transform(train_key_r_first_match)
plt.scatter(train_key_r_first_PCA[:,0], train_key_r_first_PCA[:,1], s=marker_size, color=all_colors[3])
train_key_r_second_PCA = pca.transform(train_key_r_second_match)
plt.scatter(train_key_r_second_PCA[:,0], train_key_r_second_PCA[:,1], s=marker_size, color=all_colors[4])
train_key_r_third_PCA = pca.transform(train_key_r_third_match)
plt.scatter(train_key_r_third_PCA[:,0], train_key_r_third_PCA[:,1], s=marker_size, color=all_colors[5])
plt.xlabel('1st PC')
plt.ylabel('2nd PC')
plt.xlim(key_w_xlim)	# Set axes to same scale as key_w to facilitate comparison
plt.ylim(key_w_ylim)
ax.legend(legend_handle, legend_labels, ncol = 3, handletextpad = -2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plot_fname = './key_r_t123_train_vs_test.png'
plt.savefig(plot_fname)
plt.close()

