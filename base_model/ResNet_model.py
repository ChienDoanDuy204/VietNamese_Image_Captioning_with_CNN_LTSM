import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset
from torchinfo import summary
from tqdm import tqdm
from MLP import BaseMLP, ResidualBlock
from sklearn.model_selection import train_test_split

class ResidualNetworkBase(BaseMLP):
    def __init__(self):
        super().__init__()
    def predict(self, X):
        with torch.no_grad():
            self.model.eval()
            X = X.to(self.device)
            logits = self.forward(X)
            return torch.argmax(logits, dim=1)
    def compute_loss(self, logits, y):
        return self.criterion(logits,y)
    def get_accuracy(self, logits, y):
        try:
            return torch.mean((torch.argmax(logits,dim =1) == y).float())
        except:
            return torch.mean((torch.argmax(logits,dim =1) == torch.argmax(y, dim = 1)).float())

# code add each element of model ResNet18
class ResNet18(ResidualNetworkBase):
    def __init__(self, num_classes:int = 1000):
        super().__init__()

############ ------- Built architecture ResNET-18 ------- ################       
        # block1
        block1 = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        self.Add_layer(block1)

        # Residual block1
        block1_sequential = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', stride = 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(num_features=64),
        )
        downsample1 = None
        residual_block1 = ResidualBlock(block=block1_sequential,downsample=downsample1)
        self.Add_layer([residual_block1, nn.ReLU(inplace=True)])

        # Residual block2
        block2_sequential = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', stride = 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(num_features=64),
        )
        downsample2 = None
        residual_block2 = ResidualBlock(block=block2_sequential,downsample=downsample2)
        self.Add_layer([residual_block2, nn.ReLU(inplace=True)])

        # Projection shortcut 3
        block3_sequential = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride = 2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(num_features=128),
        )
        downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride = 2),
            nn.BatchNorm2d(num_features=128)
        )
        residual_block3 = ResidualBlock(block=block3_sequential,downsample=downsample3)
        self.Add_layer([residual_block3, nn.ReLU(inplace=True)])

        # Residual block 4
        block4_sequential = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same', stride = 1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(num_features=128),
        )
        downsample4 = None
        residual_block4 = ResidualBlock(block=block4_sequential,downsample=downsample4)
        self.Add_layer([residual_block4, nn.ReLU(inplace=True)])

        # project shortcut 5
        block5_sequential = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=256)
        )
        downsample5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256 ,kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256)
        )
        residual_block5 = ResidualBlock(block=block5_sequential, downsample= downsample5)
        self.Add_layer([residual_block5,nn.ReLU(inplace=True)])

        # residual block 6
        block6_sequential = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=256)
        )
        downsample6 = None
        residual_block6 = ResidualBlock(block=block6_sequential, downsample= downsample6)
        self.Add_layer([residual_block6,nn.ReLU(inplace=True)])


        # project shortcut 7
        block7_sequential = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=512)
        )
        downsample7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512 ,kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=512)
        )
        residual_block7 = ResidualBlock(block=block7_sequential, downsample= downsample7)
        self.Add_layer([residual_block7,nn.ReLU(inplace=True)])

        # residual block 8
        block8_sequential = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=512)
        )
        downsample8 = None
        residual_block8 = ResidualBlock(block=block8_sequential, downsample= downsample8)
        self.Add_layer([residual_block8,nn.ReLU(inplace=True)])

        # average pooling
        block9 = [
            nn.AvgPool2d(kernel_size=7),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes)
        ]
        self.Add_layer(block9)



# code genalization of model ResNet
class ResNet(ResidualNetworkBase):
    def __init__(self, version: int = 18, num_classes = None):
        super().__init__()
        self.list_block = None
        self.bottleneck_block = False
        # ResNet stem - Hai layer cố định của mạng ResNet
        self.Add_layer([
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2 ,padding=1)
        ])
        self.version = version
        self.num_classes = num_classes
        self.list_channel = None
        if self.num_classes is None:
            raise ValueError("num_classes must be interger!")

        ###### ResNet18 #######
        if self.version == 18:
            self.list_channel = [64,128, 256, 512]
            self.list_rsblock =[2,2,2,2]
            for i in range (len(self.list_rsblock)):
                rsblock = self.list_rsblock[i]
                for j in range (rsblock):
                    stride = 2 if (i and j ==0) else 1
                    if not i:
                        in_channel = 64
                        out_channel = self.list_channel[i]
                    if not j and i:
                        in_channel = self.list_channel[i-1]
                        out_channel = self.list_channel[i]
                    if j and i:
                        in_channel = self.list_channel[i]
                        out_channel = self.list_channel[i]
                    self.Add_layer([self.built_block_residual(in_channel=in_channel, out_channel=out_channel,  stride=stride), nn.ReLU(inplace=True)])
        ###### ResNet34 #####
        if self.version == 34:
            self.list_channel = [64,128, 256, 512]
            self.list_rsblock =[3,4,6,3]
            for i in range (len(self.list_rsblock)):
                rsblock = self.list_rsblock[i]  
                for j in range(rsblock):
                    stride = 2 if (i and j ==0) else 1
                    if not i:
                        in_channel = 64
                        out_channel = self.list_channel[i]
                    if not j and i:
                        in_channel = self.list_channel[i-1]
                        out_channel = self.list_channel[i]
                    if j and i:
                        in_channel = self.list_channel[i]
                        out_channel = self.list_channel[i]
                    self.Add_layer([self.built_block_residual(in_channel=in_channel, out_channel=out_channel, stride=stride), nn.ReLU(inplace=True)])
        ###### ResNet50 #####
        if self.version == 50:
            self.bottleneck_block = True
            self.list_channel = [256, 512, 1024, 2048]
            self.list_channel_pointwise = [64, 128, 256, 512]
            self.list_rsblock = [3, 4, 6, 3]
            for i in range(len(self.list_rsblock)):
                rsblock = self.list_rsblock[i]
                for j in range(rsblock):
                    stride = 2 if (i and j ==0) else 1
                    if not i and not j:
                        in_channel = 64
                        out_channel = self.list_channel[i]
                        out_channel_pointwise = self.list_channel_pointwise[i]
                    if j:
                        in_channel = self.list_channel[i]
                        out_channel = self.list_channel[i]
                        out_channel_pointwise = self.list_channel_pointwise[i]
                    if not j and i:
                        in_channel = self.list_channel[i-1]
                        out_channel = self.list_channel[i]
                        out_channel_pointwise = self.list_channel_pointwise[i]
                    self.Add_layer([self.built_block_residual(in_channel=in_channel, out_channel=out_channel, out_channel_pointwise=out_channel_pointwise, stride=stride), nn.ReLU(inplace=True)])
        ###### ResNet101 ####
        if self.version == 101:
            self.bottleneck_block = True
            self.list_channel = [256, 512, 1024, 2048]
            self.list_channel_pointwise = [64, 128, 256, 512]
            self.list_rsblock = [3, 4, 23, 3]
            for i in range(len(self.list_rsblock)):
                rsblock = self.list_rsblock[i]
                for j in range(rsblock):
                    stride = 2 if (i and j ==0) else 1
                    if not i and not j:
                        in_channel = 64
                        out_channel = self.list_channel[i]
                        out_channel_pointwise = self.list_channel_pointwise[i]
                    if j:
                        in_channel = self.list_channel[i]
                        out_channel = self.list_channel[i]
                        out_channel_pointwise = self.list_channel_pointwise[i]
                    if not j and i:
                        in_channel = self.list_channel[i-1]
                        out_channel = self.list_channel[i]
                        out_channel_pointwise = self.list_channel_pointwise[i]
                    self.Add_layer([self.built_block_residual(in_channel=in_channel, out_channel=out_channel, out_channel_pointwise=out_channel_pointwise, stride = stride), nn.ReLU(inplace=True)])
        ###### ResNet152 ####
        if self.version == 152:
            self.bottleneck_block = True
            self.list_channel = [256, 512, 1024, 2048]
            self.list_channel_pointwise = [64, 128, 256, 512]
            self.list_rsblock = [3, 8, 36, 3]
            for i in range(len(self.list_rsblock)):
                rsblock = self.list_rsblock[i]
                for j in range(rsblock):
                    stride = 2 if (i and j ==0) else 1
                    if not i and not j:
                        in_channel = 64
                        out_channel = self.list_channel[i]
                        out_channel_pointwise = self.list_channel_pointwise[i]
                    if j:
                        in_channel = self.list_channel[i]
                        out_channel = self.list_channel[i]
                        out_channel_pointwise = self.list_channel_pointwise[i]
                    if not j and i:
                        in_channel = self.list_channel[i-1]
                        out_channel = self.list_channel[i]
                        out_channel_pointwise = self.list_channel_pointwise[i]
                    self.Add_layer([self.built_block_residual(in_channel=in_channel, out_channel=out_channel, out_channel_pointwise=out_channel_pointwise, stride = stride), nn.ReLU(inplace=True)])

        #avg- Layer and FC
        self.Add_layer([
            nn.AvgPool2d(kernel_size=7),
            nn.Flatten(),
            nn.Linear(in_features=self.list_channel[-1], out_features=self.num_classes)
        ])
        
    def built_block_residual(self, in_channel, out_channel, out_channel_pointwise = None, stride = 1):
        if not self.bottleneck_block:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=2), nn.BatchNorm2d(num_features=out_channel)
            ) if stride != 1 or in_channel!= out_channel else None
            # basic block
            residual_block = ResidualBlock(
                block = nn.Sequential(
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias = True),
                    nn.BatchNorm2d(num_features=out_channel),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias = True),
                    nn.BatchNorm2d(num_features=out_channel),
                ),
                downsample=downsample
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride), nn.BatchNorm2d(num_features=out_channel)
            ) if stride != 1 or in_channel!= out_channel else None
            # bottlenech block
            residual_block = ResidualBlock(
                block = nn.Sequential(
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel_pointwise, kernel_size=1, stride=stride, bias = True),
                    nn.BatchNorm2d(num_features=out_channel_pointwise),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=out_channel_pointwise, out_channels=out_channel_pointwise, kernel_size=3, stride=1, padding=1, bias = True),
                    nn.BatchNorm2d(num_features=out_channel_pointwise),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=out_channel_pointwise, out_channels=out_channel, kernel_size=1, stride=1, bias=True),
                    nn.BatchNorm2d(num_features=out_channel),
                ),
                downsample=downsample
            )
        return residual_block