import torch
import  torch.nn as nn



class GenModel(nn.Module):
	def __init__(self, based_mod, n_classes, last_shape):
		super(GenModel, self).__init__()

		self.base = based_mod

		self.last = nn.Linear(last_shape, n_classes)

	def forward(self, x):
		x = self.base(x)
		x = self.last(x)

		return x
