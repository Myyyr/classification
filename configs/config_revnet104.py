import models.RevNet as RevNet
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

exp = 4
MODEL = RevNet.RevNet104([64*exp, 128*exp, 256*exp, 512*exp])

FOLD = "revnet104"