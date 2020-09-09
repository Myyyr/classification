import models.ResNet as ResNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3' 
import scheduler


BATCH_SIZE = 128
N_WORKERS = 4

EPOCHS = 300
PATIENCE = -1

LR = 0.1
W_DECAY = 5e-4#2e-4
MOMENTUM = 0.9#0.9


LRS = [0.1, 0.02, 0.004, 0.0008]
LR_EPOCH = [60, 120, 160, 200]

def SCHEDULER(optimizer):
	return scheduler.SimpleScheduler(optimizer, LR_EPOCH, LRS)

MODEL = ResNet.ResNet12(num_classes=100)

FOLD = "resnet12_cifar100_ssc"