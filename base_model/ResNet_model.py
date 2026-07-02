import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset
from torchinfo import summary
from tqdm import tqdm
from MLP import BaseMLP, ResidualBlock
from sklearn.model_selection import train_test_split


# code genalization of model ResNet
class ResNet(BaseMLP):
    def __init__(self, version: int = 18, num_classes = 1000):
        super().__init__()
        self.list_block = None
        self.bottleneck_block = False
        self.version = version
        self.num_classes = num_classes
        self.list_channel = None



############################################################ Architecture of ResNet #############################################################
        # ResNet stem - Hai layer cố định của mạng ResNet
        self.Add_layer([
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2 ,padding=1)
        ])



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
######################################################### End of Architecture of ResNet #########################################################

######################################################### Built Block Residual ##################################################################
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
####################################################### method ##################################################################################
    # def forward of model
    def forward(self, X):
        return super().forward(X)
    # def predict of model
    def predict(self, X):
        with torch.no_grad():
            self.model.eval()
            X = X.to(self.device)
            logits = self.forward(X)
            return torch.argmax(logits, dim=1)
    # def compute loss of model
    def compute_loss(self, logits, y):
        return self.criterion(logits,y)
    # def compute accuracy of model ResNet
    def get_accuracy(self, logits, y):
        try:
            return torch.mean((torch.argmax(logits,dim =1) == y).float())
        except:
            return torch.mean((torch.argmax(logits,dim =1) == torch.argmax(y, dim = 1)).float())