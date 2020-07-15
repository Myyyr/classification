'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import revtorch as rv

class pad(nn.Module):
    def __init__(self, size):
        super(pad, self).__init__()
        self.size = size
        self.zpad = nn.ZeroPad2d((size, size, 0,0))


    def forward(self, x):
        x = x.permute((0,3,2,1))
        x = self.zpad(x)
        x = x.permute((0,3,2,1))
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        planes = planes//self.expansion
        # in_planes = in_planes//self.expansion

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               in_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RevNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(RevNet, self).__init__()
        self.channels = [64*2, 128*2, 256*2, 512*2]
        self.in_planes = self.channels[0]

        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.layer1 = self._make_layer(block, self.channels[0] , num_blocks[0], [])
        self.layer2 = self._make_layer(block, self.channels[1], num_blocks[1], [torch.nn.AvgPool2d(2,2), pad((self.channels[1] - self.channels[0] )//2)])
        self.layer3 = self._make_layer(block, self.channels[2], num_blocks[2], [torch.nn.AvgPool2d(2,2), pad((self.channels[2] - self.channels[1])//2)])
        self.layer4 = self._make_layer(block, self.channels[3], num_blocks[3], [torch.nn.AvgPool2d(2,2), pad((self.channels[3] - self.channels[2])//2)])

        self.bn2  = nn.BatchNorm2d(self.channels[3])
        self.relu = nn.ReLU() 
        self.linear = nn.Linear(self.channels[3]*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, down):
        # strides = [stride] + [1]*(num_blocks-1)
        # layers = []
        # for stride in strides:
        #     layers.append(block(self.in_planes, planes, stride))
        #     self.in_planes = planes * block.expansion
        # return nn.Sequential(*layers)

        self.in_planes = planes
        layers = []
        for blc in range(num_blocks):
            block_in_planes = self.in_planes // 2
            fblock = block(block_in_planes, block_in_planes)
            gblock = block(block_in_planes, block_in_planes)

            layers.append(rv.ReversibleBlock(fblock, gblock))

        revseq = rv.ReversibleSequence(nn.ModuleList(layers))

        layers = down 
        layers.append(revseq)
        # layers += down
        # layers.append(torch.nn.AvgPool2d(2,2))
        # layers.append(pad((planes - self.in_planes)//2))

        

        return nn.Sequential(*layers)



    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu(self.bn2(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def RevNet18():
    return RevNet(BasicBlock, [1, 1, 1, 1])


def RevNet34():
    return RevNet(BasicBlock, [2, 2, 3, 1])


def RevNet50():
    return RevNet(Bottleneck, [3, 4, 6, 3])


def RevNet104():
    return RevNet(Bottleneck, [2, 3, 10, 2])#[3, 4, 23, 3]) # = 33


def RevNet152():
    return RevNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = RevNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
