import torch
import torch.nn as nn
import torchvision
import tools

import torch.optim as optim
import torchvision.transforms as transforms

import sys

import time
from tqdm import tqdm

import cifar10_data_loader
import train

from config_resnet34 import *


def test(net, testloader, ltestset, nClasses = 10):
	correct = 0
	total = 0
	
	with tqdm(total=ltestset, desc=f'Epoch {1}/{1}', unit='img') as pbar:
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs = inputs.cuda()
			targets = targets.cuda()

			out = net(inputs)
			_, predicted = torch.max(out.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()

			
			pbar.set_postfix(**{'Acc ': 100.*correct.numpy()/total})
			pbar.update(BATCH_SIZE)

		return 100.*correct.numpy()/total


def main():

	trainloader, validloader = cifar10_data_loader.get_train_valid_loader('./data', BATCH_SIZE, True, 1)
	testloader, ltestset = cifar10_data_loader.get_test_loader('./data', BATCH_SIZE, shuffle=False)




	nClasses = 10
	in_shape = [3, 32, 32]

	model = MODEL
	model.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=MOMENTUM, weight_decay=W_DECAY)

	train_results = train.train(model,
					          criterion,
					          optimizer,
					          EPOCHS,
					          {"train":trainloader, "dev":validloader},
					          fold_num = 1,
					          scheduler = train.get_scheduler(optimizer, MIN_LR, MAX_LR, STEPSIZE),
					          patience = 4,
					          find_lr=False)
	train.save_results({FOLD:train_results}, "./results") 

	test_results = train.train(model,
					          criterion,
					          optimizer,
					          1,
					          {"test":testloader},
					          fold_num = 1,
					          scheduler = train.get_scheduler(optimizer, MIN_LR, MAX_LR, STEPSIZE),
					          patience = 4,
					          find_lr=False)
	train.save_results({FOLD:test_results}, "./results") 

	
def convert_bytes(size, isbytes = True):
	if isbytes :
		b = 1000.0
	else:
		b = 1024.0		
	for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
	   if size < b:
		   return "%3.1f %s" % (size, x)
	   size /= b

	return size

# def checkpoints(model, loader, epoch, dataset, n = 10):
# 	if epoch % 10 ==0:
# 		f = open("checkpoints.txt", "a")
# 		f.write("Epoch:"+str(epoch)+";Acc:"+str(test(model, loader, epoch, dataset)))
# 		f.close()


if __name__ == '__main__':
	main()



