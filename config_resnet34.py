import ResNet

BATCH_SIZE = 512
EPOCHS = 100
LR = 0.1
W_DECAY = 5e-4#2e-4
N_WORKERS = 2
MOMENTUM = 0.9#0.9



MODEL = ResNet.resnet34()