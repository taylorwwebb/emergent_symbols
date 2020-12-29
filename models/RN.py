import torch
import torch.nn as nn
from util import log
import numpy as np
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
		self.z_size = 128
		# G_theta MLP
		log.info('Building G_theta MLP...')
		self.g_theta_hidden = nn.Linear((self.z_size+1)*2, 512)
		self.g_theta_out = nn.Linear(512, 256)
		# F_phi MLP
		log.info('Building F_phi MLP...')
		self.f_phi_hidden = nn.Linear(256, 256)
		self.y_out = nn.Linear(256, task_gen.y_dim)
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
		# Initialize parameters
		for name, param in self.named_parameters():
			# Encoder parameters have already been initialized
			if not ('encoder' in name):
				# Initialize all biases to 0
				if 'bias' in name:
					nn.init.constant_(param, 0.0)
				else:
					if 'weight' in name:
						if 'y_out' in name:
							# Initialize weights for multiple-choice output layer (followed by softmax) using Xavier normal distribution
							nn.init.xavier_normal_(param)
						else:
							# Initialize all other weights (followed by ReLU) using Kaiming normal distribution
							nn.init.kaiming_normal_(param, nonlinearity='relu')
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
		# Append temporal tag to all z
		temp_tag = torch.Tensor(np.arange(x_seq.shape[1])).unsqueeze(0).unsqueeze(2).repeat(x_seq.shape[0], 1, 1).to(device)
		z_seq = torch.cat([z_seq, temp_tag], 2)
		# Pass all pairs of z to G_theta
		all_g = []
		for z1 in range(z_seq.shape[1]):
			for z2 in range(z_seq.shape[1]):
				g_hidden = self.relu(self.g_theta_hidden(torch.cat([z_seq[:,z1,:], z_seq[:,z2,:]], dim=1)))
				g_out = self.relu(self.g_theta_out(g_hidden))
				all_g.append(g_out)
		# Stack and sum all outputs from G_theta
		all_g = torch.stack(all_g, 1).sum(1)
		# Apply F_phi
		f_hidden = self.relu(self.f_phi_hidden(all_g))
		# Output layers
		y_pred_linear = self.y_out(f_hidden)
		y_pred = y_pred_linear.argmax(1)
		return y_pred_linear, y_pred
	def apply_context_norm(self, z_seq):
		eps = 1e-8
		z_mu = z_seq.mean(1)
		z_sigma = (z_seq.var(1) + eps).sqrt()
		z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
		z_seq = (z_seq * self.gamma) + self.beta
		return z_seq