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
		# PrediNet parameters
		log.info('Building PrediNet...')
		self.z_size = 128
		self.z_tagged_size = self.z_size + 1
		self.key_size = 16
		self.N_heads = 32
		self.N_relations = 16
		self.W_k = nn.Linear(self.z_tagged_size, self.key_size, bias=False)
		self.W_q1 = nn.Linear(task_gen.seq_len*self.z_tagged_size, self.N_heads*self.key_size, bias=False)
		self.W_q2 = nn.Linear(task_gen.seq_len*self.z_tagged_size, self.N_heads*self.key_size, bias=False)
		self.W_s = nn.Linear(self.z_tagged_size, self.N_relations, bias=False)
		# PrediNet multiple-choice output MLP
		log.info('Building output MLP for multiple-choice...')
		self.PrediNet_out_size = self.N_heads * (self.N_relations + 2)
		self.out_hidden = nn.Linear(self.PrediNet_out_size, 8)
		self.y_out = nn.Linear(8, task_gen.y_dim)
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
		self.softmax = nn.Softmax(dim=1)
		# Initialize parameters
		for name, param in self.named_parameters():
			# Encoder parameters have already been initialized
			if not ('encoder' in name):
				# Initialize all biases to 0
				if 'bias' in name:
					nn.init.constant_(param, 0.0)
				else:
					if 'W_' in name:
						# Initialize weights for keys, queries, and relations using Xavier normal distribution
						nn.init.xavier_normal_(param)
					elif 'out_hidden' in name:
						# Initialize weights for output MLP hidden layer (followed by ReLU) using Kaiming normal distribution
						nn.init.kaiming_normal_(param, nonlinearity='relu')
					elif 'y_out' in name:
						# Initialize weights for multiple-choice output layer (followed by softmax) using Xavier normal distribution
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
		# Append temporal tag to all z
		temp_tag = torch.Tensor(np.arange(x_seq.shape[1])).unsqueeze(0).unsqueeze(2).repeat(x_seq.shape[0], 1, 1).to(device)
		z_seq = torch.cat([z_seq, temp_tag], 2)
		# Get keys for all objects in sequence
		K = self.W_k(z_seq)
		# Get queries for objects 1 and 2
		z_seq_flat = torch.flatten(z_seq, 1)
		Q1 = self.W_q1(z_seq_flat)
		Q2 = self.W_q2(z_seq_flat)
		# Reshape queries (separate heads)
		Q1_reshaped = Q1.view(-1, self.N_heads, self.key_size)
		Q2_reshaped = Q2.view(-1, self.N_heads, self.key_size)
		# Extract attended objects
		E1 = (self.softmax((Q1_reshaped.unsqueeze(1) * K.unsqueeze(2)).sum(dim=3)).unsqueeze(3) * z_seq.unsqueeze(2)).sum(dim=1)
		E2 = (self.softmax((Q2_reshaped.unsqueeze(1) * K.unsqueeze(2)).sum(dim=3)).unsqueeze(3) * z_seq.unsqueeze(2)).sum(dim=1)
		# Compute relation vector
		D = self.W_s(E1) - self.W_s(E2)
		# Add temporal position tag
		D = torch.cat([D, E1[:,:,-1].unsqueeze(2), E2[:,:,-1].unsqueeze(2)], 2)
		# Cconcatenate output of all heads
		R = D.view(-1, self.PrediNet_out_size)
		# Output MLP for multiple-choice
		out_MLP_hidden = self.relu(self.out_hidden(R))
		y_pred_linear = self.y_out(out_MLP_hidden)
		y_pred = y_pred_linear.argmax(1)
		return y_pred_linear, y_pred
	def apply_context_norm(self, z_seq):
		eps = 1e-8
		z_mu = z_seq.mean(1)
		z_sigma = (z_seq.var(1) + eps).sqrt()
		z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
		z_seq = (z_seq * self.gamma) + self.beta
		return z_seq