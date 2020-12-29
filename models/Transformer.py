import torch
import torch.nn as nn
import math
from util import log
import numpy as np
from modules import *

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)
	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return x

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
		# Positional encoding
		self.pos_encoder = PositionalEncoding(self.z_size)
		# Transformer
		log.info('Building transformer encoder...')
		self.n_heads = 8
		self.dim_ff = 512
		self.n_layers = 1
		self.encoder_layers = nn.TransformerEncoderLayer(self.z_size, self.n_heads, dim_feedforward=self.dim_ff, dropout=0.0, activation='relu')
		self.encoder_norm = nn.LayerNorm(self.z_size) 
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, self.n_layers, norm=self.encoder_norm)
		# Output layers
		log.info('Building output layers...')
		self.out_hidden = nn.Linear(self.z_size, 256)
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
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			else:
				# Initialize transformer parameters
				if 'transformer' in name:
					# Initialize attention weights using Xavier normal distribution
					if 'self_attn' in name:
						nn.init.xavier_normal_(param)
					# Initialize feedforward weights (followed by ReLU) using Kaiming normal distribution
					if 'linear' in name:
						nn.init.kaiming_normal_(param, nonlinearity='relu')
				# Initialize output layers
				# Initialize output hidden layer (followed by ReLU) using Kaiming normal distribution
				if 'out_hidden' in name:
					nn.init.kaiming_normal_(param, nonlinearity='relu')
				# Initialize weights for multiple-choice output layer (followed by softmax) using Xavier normal distribution
				if 'y_out' in name:
					nn.init.xavier_normal_(param)
	def forward(self, x_seq, device):
		# Encode all images in sequence
		z_seq = []
		for t in range(x_seq.shape[1]):
			x_t = x_seq[:,t,:,:].unsqueeze(1)
			z_t = self.encoder(x_t)
			z_seq.append(z_t)
		z_seq = torch.stack(z_seq, dim=0)
		if self.contextnorm:
			z_seq_all_seg = []
			for seg in range(len(self.task_seg)):
				z_seq_all_seg.append(self.apply_context_norm(z_seq[self.task_seg[seg],:,:]))
			z_seq = torch.cat(z_seq_all_seg, dim=0)
		# Positional encoding
		z_seq_pe = self.pos_encoder(z_seq)
		# Apply transformer
		z_seq_transformed = self.transformer_encoder(z_seq_pe)
		# Average over transformed sequence
		z_seq_transformed_avg = z_seq_transformed.mean(0)
		# Output layers
		out_hidden = self.relu(self.out_hidden(z_seq_transformed_avg))
		y_pred_linear = self.y_out(out_hidden)
		y_pred = y_pred_linear.argmax(1)
		return y_pred_linear, y_pred
	def apply_context_norm(self, z_seq):
		eps = 1e-8
		z_mu = z_seq.mean(0)
		z_sigma = (z_seq.var(0) + eps).sqrt()
		z_seq = (z_seq - z_mu.unsqueeze(0)) / z_sigma.unsqueeze(0)
		z_seq = (z_seq * self.gamma) + self.beta
		return z_seq