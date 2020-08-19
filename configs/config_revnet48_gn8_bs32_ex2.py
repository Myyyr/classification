import models.RevNet as RevNet
import os
import scheduler
os.environ["CUDA_VISIBLE_DEVICES"] = '3' 

BATCH_SIZE = 32
N_WORKERS = 4

EPOCHS = 300
PATIENCE = -1

LR = 0.1
W_DECAY = 5e-4#2e-4
MOMENTUM = 0.9#0.9


# LRS = [0.1,0.01,0.001]
LRS = [0.1, 0.02, 0.004, 0.0008]
LR_EPOCH = [15, 60, 150, 210]

MODEL = RevNet.RevNet48GN(group_norm=8)


# MODEL = RevNet.RevNet104()

def SCHEDULER(optimizer):
	return scheduler.SimpleScheduler(optimizer, LR_EPOCH, LRS)

FOLD = "revnet48_gn8_bs32_ex2"