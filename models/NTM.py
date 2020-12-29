import torch
import torch.nn as nn
import numpy as np
from util import log
from modules import *

class Model(nn.Module):
	def __init__(self, task_gen, args):
		super(Model, self).__init__()
		# Encoder
		log.info('Building encoder...')
		if args.encoder == 'conv':
			self.encoder = Encoder_conv(args)
		elif args.encoder == 'mlp':
			self.encoder = Encoder_mlp(args)
		elif args.encoder == 'rand':
			self.encoder = Encoder_rand(args)
		# LSTM and output layers
		log.info('Building LSTM and output layers...')
		self.z_size = 128
		self.hidden_size = 512
		self.mem_vec_size = 256
		self.lstm = nn.LSTM(self.z_size + self.mem_vec_size, self.hidden_size, batch_first=True)
		self.erase_out = nn.Linear(self.hidden_size, self.mem_vec_size)
		self.add_out = nn.Linear(self.hidden_size, self.mem_vec_size)
		self.write_key_out = nn.Linear(self.hidden_size, self.mem_vec_size)
		self.read_key_out = nn.Linear(self.hidden_size, self.mem_vec_size)
		self.y_out = nn.Linear(self.hidden_size, task_gen.y_dim)
		# Key strengths
		self.read_key_strength_out = nn.Linear(self.hidden_size, 1)
		self.write_key_strength_out = nn.Linear(self.hidden_size, 1)
		# Gates
		self.read_interp_gate_out = nn.Linear(self.hidden_size, 1)
		self.write_interp_gate_out = nn.Linear(self.hidden_size, 1)
		# Shift weights
		self.mem_size = 10
		self.allowable_shifts = [-1, 0, 1]
		self.N_allowable_shifts = len(self.allowable_shifts)
		shift_ind = []
		for shift in self.allowable_shifts:
			shift_ind.append(np.roll(np.arange(self.mem_size), shift))
		self.shift_ind = torch.Tensor(np.concatenate(shift_ind)).long()
		self.read_shift_w_out = nn.Linear(self.hidden_size, self.N_allowable_shifts)
		self.write_shift_w_out = nn.Linear(self.hidden_size, self.N_allowable_shifts)
		# Sharpening
		self.read_sharpen_out = nn.Linear(self.hidden_size, 1)
		self.write_sharpen_out = nn.Linear(self.hidden_size, 1)
		# Memory
		log.info('Initialize memory matrix as learnable parameter...')
		self.memory_init = nn.Parameter(torch.zeros(self.mem_size, self.mem_vec_size))
		# Similarity metric
		self.sim_metric = nn.CosineSimilarity(dim=2)
		# Context normalization
		if args.norm_type == 'contextnorm' or args.norm_type == 'tasksegmented_contextnorm':
			self.contextnorm = True
			self.gamma = nn.Parameter(torch.ones(self.z_size))
			self.beta = nn.Parameter(torch.zeros(self.z_size))
		else:
			self.contextnorm = False
		if args.norm_type == 'tasksegmented_contextnorm':
			self.task_seg = task_gen.task_seg
		else:
			self.task_seg = [np.arange(task_gen.seq_len)]
		# Nonlinearities
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.softplus = nn.Softplus()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)
		# Initialize parameters
		for name, param in self.named_parameters():
			# Encoder parameters have already been initialized
			if not ('encoder' in name):
				# Initialize all biases to 0
				if 'bias' in name:
					nn.init.constant_(param, 0.0)
				else:
					if 'lstm' in name:
						# Initialize gate weights (followed by sigmoid) using Xavier normal distribution
						nn.init.xavier_normal_(param[:self.hidden_size*2,:])
						nn.init.xavier_normal_(param[self.hidden_size*3:,:])
						# Initialize input->hidden and hidden->hidden weights (followed by tanh) using Xavier normal distribution with gain = 5/3
						nn.init.xavier_normal_(param[self.hidden_size*2:self.hidden_size*3,:], gain=5.0/3.0)
					elif 'key_out' in name:
						# Initialize weights for key output layers (followed by tanh) using Xavier normal distribution with gain = 5/3
						nn.init.xavier_normal_(param, gain=5.0/3.0)
					elif 'gate' in name:
						# Initialize weights for gate output layers (followed by sigmoid) using Xavier normal distribution
						nn.init.xavier_normal_(param)
					elif 'shift' in name:
						# Initialize weights for shift weight layers (followed by softmax) using Xavier normal distribution
						nn.init.xavier_normal_(param)
					elif 'sharpen' in name:
						# Initialize weights for sharpening output layers (followed by softplus) using Kaiming normal distribution
						nn.init.kaiming_normal_(param, nonlinearity='relu')
					elif 'strength' in name:
						# Initialize weights for key strength output layers (followed by softplus) using Kaiming normal distribution
						nn.init.kaiming_normal_(param, nonlinearity='relu')
					elif 'erase' in name:
						# Initialize weights for erase vector output layer (followed by sigmoid) using Xavier normal distribution
						nn.init.xavier_normal_(param)
					elif 'add' in name:
						# Initialize weights for add vector output layer (followed by tanh) using Xavier normal distribution with gain = 5/3
						nn.init.xavier_normal_(param, gain=5.0/3.0)
					elif 'y_out' in name:
						# Initialize weights for multiple-choice output layer (followed by softmax) using Xavier normal distribution
						nn.init.xavier_normal_(param)
					elif 'memory_init' in name:
						# Initialize memory matrix (followed by softmax) using Xavier normal distribution
						nn.init.xavier_normal_(param)
	def forward(self, x_seq, device):
		# Encode all images in sequence
		z_seq = []
		for t in range(x_seq.shape[1]):
			x_t = x_seq[:,t,:,:].unsqueeze(1)
			z_t = self.encoder(x_t)
			z_seq.append(z_t)
		z_seq = torch.stack(z_seq, dim=1)
		if self.contextnorm:
			z_seq_all_seg = []
			for seg in range(len(self.task_seg)):
				z_seq_all_seg.append(self.apply_context_norm(z_seq[:,self.task_seg[seg],:]))
			z_seq = torch.cat(z_seq_all_seg, dim=1)
		# Initialize hidden state
		hidden = torch.zeros(1, x_seq.shape[0], self.hidden_size).to(device)
		cell_state = torch.zeros(1, x_seq.shape[0], self.hidden_size).to(device)
		# Initialize read vector
		read_vec = torch.zeros(x_seq.shape[0], 1, self.mem_vec_size).to(device)
		# Initialize memory
		memory = self.memory_init.unsqueeze(0)
		# Initialize read and write weights
		w_read = torch.zeros(x_seq.shape[0], self.mem_size).to(device)
		w_write = torch.zeros(x_seq.shape[0], self.mem_size).to(device)
		# Memory model (extra time step to process key retrieved on final time step)
		for t in range(x_seq.shape[1] + 1):
			# Image embedding
			if t == x_seq.shape[1]:
				z_t = torch.zeros(x_seq.shape[0], 1, self.z_size).to(device)
			else:
				z_t = z_seq[:,t,:].unsqueeze(1)
			# Controller
			# LSTM
			lstm_in = torch.cat([z_t, read_vec], dim=2)
			lstm_out, (hidden, cell_state) = self.lstm(lstm_in, (hidden, cell_state))
			# Read head
			read_key = self.tanh(self.read_key_out(lstm_out))
			read_key_strength = self.softplus(self.read_key_strength_out(lstm_out).squeeze(2))
			read_interp_gate = self.sigmoid(self.read_interp_gate_out(lstm_out.squeeze()))
			read_shift_w = self.softmax(self.read_shift_w_out(lstm_out.squeeze()))
			read_sharpen = self.softplus(self.read_sharpen_out(lstm_out.squeeze())) + 1
			# Write head
			write_key = self.tanh(self.write_key_out(lstm_out))
			write_key_strength = self.softplus(self.write_key_strength_out(lstm_out).squeeze(2))
			write_interp_gate = self.sigmoid(self.write_interp_gate_out(lstm_out.squeeze()))
			write_shift_w = self.softmax(self.write_shift_w_out(lstm_out.squeeze()))
			write_sharpen = self.softplus(self.write_sharpen_out(lstm_out.squeeze())) + 1
			erase_vec = self.sigmoid(self.erase_out(lstm_out))
			add_vec = self.tanh(self.add_out(lstm_out))
			# Task output layers
			y_pred_linear = self.y_out(lstm_out).squeeze()
			y_pred = y_pred_linear.argmax(1)
			# Read from memory
			if t == 0:
				read_vec = torch.zeros(x_seq.shape[0], 1, self.mem_vec_size).to(device)
			else:
				# Read from memory
				# Cache read weights from previous time step
				w_read_prev = w_read
				# Content-based attention
				w_read = self.sim_metric(read_key, memory)
				w_read = torch.exp(read_key_strength * w_read) / torch.exp(read_key_strength * w_read).sum(1).unsqueeze(1)
				# Interpolation with read weights from previous time step
				if t > 1:
					w_read = (read_interp_gate * w_read) + ((1 - read_interp_gate) * w_read_prev)
				# Circular shift
				w_read_shifted = torch.gather(w_read, 1, self.shift_ind.to(device).unsqueeze(0).repeat(x_seq.shape[0],1)).view(-1, self.N_allowable_shifts, self.mem_size)
				w_read = (w_read_shifted * read_shift_w.unsqueeze(2)).sum(1)
				# Sharpening
				w_read_pre_sharpen = w_read
				w_read = (w_read ** read_sharpen) / (w_read ** read_sharpen).sum(1).unsqueeze(1)
				# Read
				read_vec = (memory * w_read.unsqueeze(2)).sum(1).unsqueeze(1)
			# Write to memory
			# Cache write weights from previous time step
			w_write_prev = w_write
			# Content-based attention
			w_write = self.sim_metric(write_key, memory)
			w_write = torch.exp(write_key_strength * w_write) / torch.exp(write_key_strength * w_write).sum(1).unsqueeze(1)
			# Interpolation with write weights from previous time step
			if t > 0:
				w_write = (write_interp_gate * w_write) + ((1 - write_interp_gate) * w_write_prev)
			# Circular shift
			w_write_shifted = torch.gather(w_write, 1, self.shift_ind.to(device).unsqueeze(0).repeat(x_seq.shape[0],1)).view(-1, self.N_allowable_shifts, self.mem_size)
			w_write = (w_write_shifted * write_shift_w.unsqueeze(2)).sum(1)
			# Sharpening
			w_write_pre_sharpen = w_write
			w_write = (w_write ** write_sharpen) / (w_write ** write_sharpen).sum(1).unsqueeze(1)
			# Erase
			memory = memory * (1 - (erase_vec * w_write.unsqueeze(2)))
			# Add
			memory = memory + (add_vec * w_write.unsqueeze(2))
		return y_pred_linear, y_pred
	def apply_context_norm(self, z_seq):
		eps = 1e-8
		z_mu = z_seq.mean(1)
		z_sigma = (z_seq.var(1) + eps).sqrt()
		z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
		z_seq = (z_seq * self.gamma) + self.beta
		return z_seq