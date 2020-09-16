import models.RevNet as RevNet
import os
import scheduler
os.environ["CUDA_VISIBLE_DEVICES"] = '3' 

BATCH_SIZE = 128
N_WORKERS = 4

EPOCHS = 600
PATIENCE = -1

LR = 0.1
W_DECAY = 5e-4#2e-4
MOMENTUM = 0.9#0.9


# LRS = [0.1,0.01,0.001]
LRS = [0.1, 0.02, 0.004, 0.0008]
LR_EPOCH = [60*2, 120*2, 160*2, 200*2]

MODEL = RevNet.RevNet162()


# MODEL = RevNet.RevNet104()

def SCHEDULER(optimizer):
	return scheduler.SimpleScheduler(optimizer, LR_EPOCH, LRS)

FOLD = "revnet162_longer_training"