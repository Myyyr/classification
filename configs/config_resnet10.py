import models.ResNet as ResNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3' # 3 > c2
# 2 > c1 
import scheduler
# import torch.optim as optim

BATCH_SIZE = 128
N_WORKERS = 2

EPOCHS = 300
#PATIENCE = 30 
# PATIENCE = 5
PATIENCE = -1 

LR = 0.1
W_DECAY = 5e-4#2e-4
MOMENTUM = 0.9#0.9


# LRS = [1,0.1, 0.01, 0.001, 0.0001] # fold = resnet18_morelr
# LRS = [0.0001, 0.001, 0.01, 0.1, 0.01, 0.001, 0.0001]

LRS = [0.1, 0.02, 0.004, 0.0008]
LR_EPOCH = [60, 120, 160, 200]

def SCHEDULER(optimizer):
	#return scheduler.MoreSimpleScheduler(optimizer, LRS)
	return scheduler.SimpleScheduler(optimizer, LR_EPOCH, LRS)
	# return optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

MODEL = ResNet.ResNet10()

FOLD = "resnet10"