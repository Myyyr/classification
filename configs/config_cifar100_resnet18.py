import models.ResNet as ResNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2' 


BATCH_SIZE = 64
N_WORKERS = 4

EPOCHS = 350
PATIENCE = 30

LR = 0.1
W_DECAY = 5e-4#2e-4
MOMENTUM = 0.9#0.9


LRS = [0.1, 0.02, 0.004, 0.0008]

MODEL = ResNet.ResNet18(num_classes=100)

FOLD = "resnet18_cifar100"