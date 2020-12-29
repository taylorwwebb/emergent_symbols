import torch
import torch.nn as nn
from util import log
import numpy as np
from modules import *

class Model(nn.Module):

	def __init__(self, task_gen, args, n_batch_mem = 1):
		super(Model, self).__init__()
		# Encoder
		log.info('Building encoder...')
		if args.encoder == 'conv':
			self.encoder = Encoder_conv(args)
		elif args.encoder == 'mlp':
			self.encoder = Encoder_mlp(args)
		elif args.encoder == 'rand':
			self.encoder = Encoder_rand(args)
		# MNM and output layers
		log.info('Building MNM and output layers...')
		self.z_size = 128
		self.mnm = MNMp(self.z_size, n_in_mem = self.z_size, n_units_mem = self.z_size, n_batch_mem = n_batch_mem)
		self.y_out = nn.Linear(self.z_size, task_gen.y_dim)
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
	def reset_state(self):
		self.mnm.reset_state()
	def forward(self, x_seq, device):
		self.reset_state()
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
		# Initialize loss
		const_loss_all = 0
		# Memory model (extra time step to process key retrieved on final time step)
		for t in range(x_seq.shape[1] + 1):
			# Image embedding
			if t == x_seq.shape[1]:
				z_t = torch.zeros(x_seq.shape[0], self.z_size).to(device)
			else:
				z_t = z_seq[:,t,:]
			# Run memory model
			h_t, const_loss, re_const_loss_init = self.mnm(z_t, device)
			# Meta objective
			const_loss_all += const_loss
		# Average meta objective over sequence
		const_loss_all /= x_seq.shape[1]
		# MC pred
		y_pred_linear = self.y_out(h_t)
		y_pred = y_pred_linear.argmax(1)
		return y_pred_linear, y_pred, const_loss_all
	def apply_context_norm(self, z_seq):
		eps = 1e-8
		z_mu = z_seq.mean(1)
		z_sigma = (z_seq.var(1) + eps).sqrt()
		z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
		z_seq = (z_seq * self.gamma) + self.beta
		return z_seq

class MNMp(nn.Module):
	def __init__(self, n_units, n_in_mem = 50, n_units_mem = 150, n_batch_mem=16, layer_norm_kv=False):
		if n_in_mem is None:
			n_in_mem = n_units
		if n_units_mem is None:
			n_units_mem = n_units
		super(MNMp, self).__init__()
		self.lstm_l1 = nn.LSTMCell(n_units+n_in_mem, n_units)
		self.heads_l2 = nn.Linear(n_units, (n_batch_mem*3+1)*n_in_mem)
		self.memfunc = FFMemoryLearned(n_in = n_in_mem, n_out = n_in_mem, n_units = n_units_mem, h_nonlinty='tanh')
		self.f_l = nn.Linear(n_in_mem, 1)
		self.read_out = nn.Linear(n_units+n_in_mem, n_units)
		self.layer_norm = nn.LayerNorm((n_batch_mem*3+1)*n_in_mem)
		self.n_batch_mem = n_batch_mem
		self.n_in_mem = n_in_mem
		self.n_units_mem = n_units_mem
		self.n_units = n_units
		self.h = None
		self.replay_k = []
		self.replay_v = []
		self.layer_norm_kv = layer_norm_kv
		self.h_lstm = None
		self.c_lstm = None
	def unchain_state(self):
		if self.h is not None:
			self.h = self.h.detach()
			self.h_lstm = self.h_lstm.detach()
			self.c_lstm = self.c_lstm.detach()
		self.replay_k[:] = []
		self.replay_v[:] = []
		self.memfunc.unchain_state()
	def reset_state(self):
		self.h = None
		self.replay_k[:] = []
		self.replay_v[:] = []
		self.memfunc.reset_mem()
		self.h_lstm = None
		self.c_lstm = None		
	def forward(self, x, device):
		if self.h is None:
			self.h = torch.zeros((x.shape[0], self.n_in_mem)).to(device).float()
			self.h_lstm = torch.zeros((x.shape[0], self.n_units)).to(device).float()
			self.c_lstm = torch.zeros((x.shape[0], self.n_units)).to(device).float()
		self.h_lstm, self.c_lstm = self.lstm_l1(torch.cat([x, self.h], dim=1), (self.h_lstm, self.c_lstm))
		if self.layer_norm_kv:
			h = self.layer_norm(self.heads_l2(self.h_lstm))
		else:
			h = torch.tanh(self.heads_l2(self.h_lstm))
		betta, n_k_v = torch.split(h, [self.n_in_mem, h.shape[1] - self.n_in_mem], dim=1)
		betta = torch.sigmoid(self.f_l(betta))		
		n_k_v = n_k_v.view(n_k_v.shape[0], self.n_batch_mem, -1).contiguous()
		k_w, v_w, k_r = torch.chunk(n_k_v, 3, dim=2)		
		re_const_loss, re_const_loss_init = self.memfunc.update(k_w, v_w, device, f_gate=betta)		
		self.h = self.memfunc.read(k_r, device)		
		h_lstm = self.read_out(torch.cat([self.h_lstm, self.h], dim=1))
		return h_lstm, re_const_loss, re_const_loss_init

class FFMemoryLearned(nn.Module):
	def __init__(self, n_in, n_out, n_units = 100, h_nonlinty='tanh', o_nonlinty='identity', read_feat=False):
		super(FFMemoryLearned, self).__init__()       
		self.l1 = nn.Linear(n_in, n_units).weight.data
		self.l2 = nn.Linear(n_units, n_units).weight.data
		self.l3 = nn.Linear(n_units, n_out).weight.data
		self.targets_net = nn.Linear(n_units, 3*n_units)
		self.feat_proj = nn.Linear(n_units+n_out, n_out)
		self.n_in = n_in
		self.n_units = n_units
		self.n_out = n_out       
		self.h_nonlinty = h_nonlinty
		self.o_nonlinty = o_nonlinty
		self.read_feat = read_feat
		self.Ws = [self.l1, self.l2, self.l3]
	def unchain_state(self):
		if self.Ws_temp is not None:
			Ws_temp = []
			for W_l in self.Ws_temp:
				Ws_temp.append(W_l.detach())
			self.Ws_temp[:] = Ws_temp
	def reset_mem(self):
		for W_l in self.Ws:
			W_l.detach()
		self.Ws_temp = None  
	def shape_input(self, x, y=None):
		if len(x.shape) > 2:
			return x, y
		x = x.unsqueeze(dim=1)
		if y is not None:
			y = y.unsqueeze(dim=1)
		return x, y
	def non_linty_pass(self, nonlinty, h):
		if nonlinty == 'tanh':
			return torch.tanh(h)
		elif nonlinty == 'sigmoid':
			return torch.sigmoid(h)
		else:
			return h
	def grad_nonlinty(self, nonlinty, h):
		if nonlinty == 'tanh':
			return 1.0 - h.pow(2)
		elif nonlinty == 'sigmoid':
			return h*(1.0 - h)
		else:
			return 1.0
	def forward(self, x, device):
		if self.Ws_temp is None:
			self.Ws_temp = []
			for W_l in self.Ws:
				W = W_l.unsqueeze(dim=0).expand((x.shape[0], W_l.shape[0], W_l.shape[1]))
				self.Ws_temp.append(W)
		h_acts = []
		h = x
		for W_l in self.Ws_temp[:-1]:
			h = torch.matmul(h, W_l.to(device).transpose(1,2))
			h = self.non_linty_pass(self.h_nonlinty, h)
			h_acts.append(h)
		y_pred = torch.matmul(h, self.Ws_temp[-1].to(device).transpose(1,2))
		y_pred = self.non_linty_pass(self.o_nonlinty, y_pred)
		return y_pred, h_acts
	def mse_loss(self, y_pred, y):
		diff = y_pred - y
		diff = diff.view(-1)
		mse = diff.dot(diff)/diff.size()[0]
		return mse
	def read(self, k, device, weights=None, avg=True):
		k, _ = self.shape_input(k)
		v, h_acts = self.forward(k, device)
		if self.read_feat:
			v = torch.cat([v, h_acts[-1]], dim=2)
			v = self.feat_proj(v)
		if weights is not None:
			v *= weights
		if avg:
			v = v.mean(dim=1)
		return v
	def update(self, x, y, device, f_gate=0.1):
		x, y = self.shape_input(x, y)
		y_pred, h_acts = self.forward(x, device)
		y_pred = y_pred.contiguous()
		y = y.contiguous()
		re_const_loss_init = self.mse_loss(y_pred.view(-1, y_pred.shape[2]), y.view(-1, y.shape[2]))
		e = self.targets_net(y.view(-1, y.shape[2]))
		e = e.view(y.shape[0], y.shape[1], -1)
		e = torch.chunk(e, 3, dim=2)
		if len(f_gate.shape) < 3:
			f_gate = f_gate.unsqueeze(1)
		h_acts_tail = h_acts + [y_pred]
		h_acts_head = [x] + h_acts
		Ws_t = []
		for W_l, pred, h, target in reversed(list(zip(self.Ws_temp, h_acts_tail, h_acts_head, e))):
			h = h*f_gate.expand(h.shape)
			diff = pred - target
			diff = diff*(2./ (diff.shape[1]*diff.shape[2]))
			W_l = W_l.to(device) - torch.matmul(diff.transpose(1,2), h)#- 0.0001*W_l 0.1
			Ws_t.insert(0, W_l)
		self.Ws_temp[:] = Ws_t
		y_pred, h_acts = self.forward(x, device)
		re_const_loss = self.mse_loss(y_pred.view(-1, y_pred.shape[2]), y.view(-1, y.shape[2]))
		return re_const_loss, re_const_loss_init