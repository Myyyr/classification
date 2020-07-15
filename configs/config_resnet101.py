import models.ResNet as ResNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 


BATCH_SIZE = 128
N_WORKERS = 4

EPOCHS = 350
PATIENCE = 30

LR = 0.1
W_DECAY = 5e-4#2e-4
MOMENTUM = 0.9#0.9


LRS = [0.1,0.01,0.001]

MODEL = ResNet.ResNet101()

FOLD = "resnet101"