import ResNet
from GenModel import GenModel

BATCH_SIZE = 512
N_WORKERS = 2

EPOCHS = 100
PATIENCE = 20

LR = 0.1
W_DECAY = 5e-4#2e-4
MOMENTUM = 0.9#0.9

MIN_LR = LR
MAX_LR = LR/100
STEPSIZE = int((0.9*50000/512))*35
GAMMA = 0.1

MODEL = ResNet.resnet34(num_classes=10)

FOLD = "resnet34-StepLR-st35g01"