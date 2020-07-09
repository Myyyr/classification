import torch
import numpy as np




class SimpleScheduler():

	def __init__(self, optimizer, epochs, lrs, iter_per_epoch):
		super(SimpleScheduler, self).__init__()

		if len(epochs) != len(lrs):
			raise ValueError("SimpleScheduler lrs and epoch not same size")

		self.n_steps = len(epochs)

		self.Optimizer = optimizer
		self.epochs = epochs
		self.lrs = lrs
		self.iter_per_epoch = iter_per_epoch

		self.iter = 0
		self.ep = 0

	def update_epoch(self):
		if self.iter == self.iter_per_epoch:
			self.iter == 0
			self.ep += 1

	def update_lr(self):
		for i in range(self.n_steps):
			if self.ep+1 == self.epochs[i]:
				self.Optimizer.param_groups[0]['lr'] = self.lrs[i]

	def step(self):
		self.update_lr()
		self.update_epoch()

		self.iter += 1
		



class MoreSimpleScheduler():

	def __init__(self, optimizer, lrs):
		super(MoreSimpleScheduler, self).__init__()

		self.Optimizer = optimizer
		self.lrs = lrs

		self.iter = 0

		self.Optimizer.param_groups[0]['lr'] = self.lrs[self.iter]
		self.end = False


	def update_lr(self):
		self.iter += 1


		if self.iter >= len(self.lrs)-1:
			self.end = True # self.end = True if it's the last learning rate to use.

	def step(self):
		self.Optimizer.param_groups[0]['lr'] = self.lrs[self.iter]
