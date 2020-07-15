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
import scheduler

import torch.backends.cudnn as cudnn


# from configs.config_revnet18 import *

def get_n_parameters(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params



def main():

	trainloader, validloader = cifar10_data_loader.get_train_valid_loader('./data', BATCH_SIZE, True, 1, 0.2)
	testloader, ltestset = cifar10_data_loader.get_test_loader('./data', BATCH_SIZE, shuffle=False)




	nClasses = 10
	in_shape = [3, 32, 32]

	model = MODEL
	print("Number of model's parameter :", get_n_parameters(model))
	device = 'cuda'
	model.to(device)
	model = torch.nn.DataParallel(model)
	cudnn.benchmark = True
	# model.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=MOMENTUM, weight_decay=W_DECAY)

	train_results = train.train(model,
					          criterion,
					          optimizer,
					          EPOCHS,
					          {"train":trainloader, "dev":validloader},
					          fold_num = FOLD,
					          scheduler = scheduler.MoreSimpleScheduler(optimizer = optimizer, lrs = LRS), #torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEPSIZE, gamma=GAMMA),#train.get_scheduler(optimizer, MIN_LR, MAX_LR, STEPSIZE),
					          patience = PATIENCE,
					          LR = LR,
					          MOMENTUM = MOMENTUM,
					          W_DECAY = W_DECAY,
					          find_lr=False)
	train.save_results({FOLD:train_results}, "./results") 

	test_results = train.train(model,
					          criterion,
					          optimizer,
					          1,
					          {"test":testloader},
					          fold_num = FOLD,
					          scheduler = scheduler.MoreSimpleScheduler(optimizer = optimizer, lrs = LRS), #torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1),#train.get_scheduler(optimizer, MIN_LR, MAX_LR, STEPSIZE),
					          patience = 4,
					          LR =LR,
					          MOMENTUM =MOMENTUM,
					          W_DECAY =W_DECAY,
					          find_lr=False,
					          results=train_results)
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


def import_config(path):
	path = path.replace("/",".")
	path = path.replace(".py","")
	module = __import__(path, fromlist=['*'])
	if hasattr(module, '__all__'):
	    all_names = module.__all__
	else:
	    all_names = [name for name in dir(module) if not name.startswith('_')]

	globals().update({name: getattr(module, name) for name in all_names})


if __name__ == '__main__':
	import  argparse

	parser = argparse.ArgumentParser(description='CNN Classif Training Function')

	parser.add_argument('-c', '--config',  help='training config file', required=True)

	import_config(parser.parse_args().config)

	main()


