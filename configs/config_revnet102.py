import models.RevNet as RevNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 3 > c2
# 2 > c1 
import scheduler

BATCH_SIZE = 128
N_WORKERS = 4

EPOCHS = 350
#PATIENCE = 30 
# PATIENCE = 5
PATIENCE = 30

LR = 0.1
W_DECAY = 5e-4#2e-4
MOMENTUM = 0.9#0.9


LRS = [0.1, 0.01, 0.001]
def SCHEDULER(optimizer):
	return scheduler.MoreSimpleScheduler(optimizer, LRS)
	

MODEL = RevNet.RevNet102bsc()

FOLD = "revnet102"