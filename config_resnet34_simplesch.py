import ResNet
from GenModel import GenModel

BATCH_SIZE = 512
N_WORKERS = 2

EPOCHS = 350
PATIENCE = -1

LR = 0.1
W_DECAY = 5e-4#2e-4
MOMENTUM = 0.9#0.9


LRS = [0.1,0.01,0.001]
SCH_EPOCHS = [0,150,250]
ITER_PER_EPOCH = int((0.9*50000/512))

MODEL = ResNet.resnet34(num_classes=10)

FOLD = "resnet34-StepLR-simplesch"