import models.ResNet as ResNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 


BATCH_SIZE = 32
N_WORKERS = 4

EPOCHS = 350
PATIENCE = -1

LR = 0.1
W_DECAY = 5e-4#2e-4
MOMENTUM = 0.9#0.9


MODEL = ResNet.ResNet18(num_classes=20)

FOLD = "resnet18_pascalvoc"