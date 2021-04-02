import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
        
from collections import OrderedDict
import torchaudio
import torch.nn as nn

#for vgg like network
cfgs = {'A': [8, 'M1', 16,  "M1",  32, 32, 'M2', 64, 64, 'M3', 128, 128, 'M3', 256, 256, 'M3', 512, 512, 'M3', 1024, 1024, 'M3', ]}



class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)
    

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
        

"""
 Implementation of EnvNet2 [Tokozume and Harada, 2017]
 opt.fs = 44000
 opt.inputLength = 66650
 Layer ksize stride # of filters Data shape
 Input (1, 1, 66,650)
 conv1 (1, 64) (1, 2) 32
 conv2 (1, 16) (1, 2) 64
 pool2 (1, 64) (1, 64) (64, 1, 260)
 swapaxes (1, 64, 260)
 conv3, 4 (8, 8) (1, 1) 32
 pool4 (5, 3) (5, 3) (32, 10, 82)
 conv5, 6 (1, 4) (1, 1) 64
 pool6 (1, 2) (1, 2) (64, 10, 38)
 conv7, 8 (1, 2) (1, 1) 128
 pool8 (1, 2) (1, 2) (128, 10, 18)
 conv9, 10 (1, 2) (1, 1) 256
 pool10 (1, 2) (1, 2) (256, 10, 8)
 fc11 - - 4096 (4,096,)
 fc12 - - 4096 (4,096,)
 fc13 - - # of classes (# of classes,)
# """

class EnvReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(EnvReLu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.layer(input)


class EnvNet2(nn.Module):
    def __init__(self, n_classes):
        super(EnvNet2, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', EnvReLu(in_channels=1,
                              out_channels=32,
                              kernel_size=(1, 64),
                              stride=(1, 2),
                              padding=0)),
            ('conv2', EnvReLu(in_channels=32,
                              out_channels=64,
                              kernel_size=(1, 16),
                              stride=(1, 2),
                              padding=0)),
            ('max_pool2', nn.MaxPool2d(kernel_size=(1, 64),
                                       stride=(1, 64),
                                       ceil_mode=True)),
            ('transpose', Transpose()),
            ('conv3', EnvReLu(in_channels=1,
                              out_channels=32,
                              kernel_size=(8, 8),
                              stride=(1, 1),
                              padding=0)),
            ('conv4', EnvReLu(in_channels=32,
                              out_channels=32,
                              kernel_size=(8, 8),
                              stride=(1, 1),
                              padding=0)),
            ('max_pool4', nn.MaxPool2d(kernel_size=(5, 3),
                                       stride=(5, 3),
                                       ceil_mode=True)),
            ('conv5', EnvReLu(in_channels=32,
                              out_channels=64,
                              kernel_size=(1, 4),
                              stride=(1, 1),
                              padding=0)),
            ('conv6', EnvReLu(in_channels=64,
                              out_channels=64,
                              kernel_size=(1, 4),
                              stride=(1, 1),
                              padding=0)),
            ('max_pool6', nn.MaxPool2d(kernel_size=(1, 4),
                                       stride=(1, 4),
                                       ceil_mode=True)),
            ('conv7', EnvReLu(in_channels=64,
                              out_channels=128,
                              kernel_size=(1, 2),
                              stride=(1, 1),
                              padding=0)),
            ('conv8', EnvReLu(in_channels=128,
                              out_channels=128,
                              kernel_size=(1, 2),
                              stride=(1, 1),
                              padding=0)),
            ('max_pool8', nn.MaxPool2d(kernel_size=(1, 4),
                                       stride=(1, 4),
                                       ceil_mode=True)),
            ('conv9', EnvReLu(in_channels=128,
                              out_channels=256,
                              kernel_size=(1, 2),
                              stride=(1, 1),
                              padding=0)),
            ('conv10', EnvReLu(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 2),
                               stride=(1, 1),
                               padding=0)),
            ('max_pool10', nn.MaxPool2d(kernel_size=(1, 2),
                                        stride=(1, 2),
                                        ceil_mode=True)),
            ('flatten', Flatten()),
            
            ]))
                                   
        self.model_cl = nn.Sequential(OrderedDict([                         
            ('fc11', nn.Linear(in_features=12800, out_features=4096, bias=True)),
            ('relu11', nn.ReLU()),
            ('dropout11', nn.Dropout()),
            ('fc12', nn.Linear(in_features=4096, out_features=4096, bias=True)),
            ('relu12', nn.ReLU()),
            ('dropout12', nn.Dropout()),
            ('fc13', nn.Linear(in_features=4096, out_features=n_classes, bias=True)),
        ]))

    def forward(self, x):
            x =  self.model(x)
            #print(x.shape)
            x = self.model_cl(x)
            return x
                                   



    
    
def VGG_block(cnf):
    layers: List[nn.Module] = []
    in_channels = 1
    for v in cnf:
        if v == "M1":
            layers += [nn.MaxPool1d(8)]
        elif v == "M2":
            layers += [nn.MaxPool1d(4)]
        elif v == "M3":
            layers += [nn.MaxPool1d(2)]
        else:
            v = cast(int, v)
            conv1d = nn.Conv1d(in_channels, v, kernel_size = 9,)
            layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
class Wavform1dVGG(nn.Module):
    def __init__(self, features, n_output=24):
        super(Wavform1d, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(19456, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_output),
        )

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.classifier(x)
        return x


def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)

# M5
class M5Net(nn.Module):
    def __init__(self):
        super(M5Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 48, 9, 4)
        self.bn1 = nn.BatchNorm1d(48)
        self.pool1 = nn.MaxPool1d(8)
        self.conv2 = nn.Conv1d(48, 192, 9)
        self.bn2 = nn.BatchNorm1d(192)
        self.pool2 = nn.MaxPool1d(8)
        self.conv3 = nn.Conv1d(192, 384, 9)
        self.bn3 = nn.BatchNorm1d(384)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(384, 768, 9)
        self.bn4 = nn.BatchNorm1d(768)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(768, 24)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        #print(x.shape)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return x


# M3
class M3Net(nn.Module):
    def __init__(self):
        super(M3Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, 80, 4)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(256, 256, 3)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(498) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(256, 24)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return x


# M11
class M11Net(nn.Module):
    def __init__(self):
        super(M11Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 80, 4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(8)
        
        self.conv2 = nn.Conv1d(64, 128, 9) 
        self.conv21 = nn.Conv1d(128, 128, 3) 
        self.bn2 = nn.BatchNorm1d(128)
        self.bn21 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(128, 256, 9)
        self.conv31 = nn.Conv1d(256, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn31 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        
        self.conv4 = nn.Conv1d(256, 512, 9)
        self.conv41 = nn.Conv1d(512, 512, 3)
        self.conv42 = nn.Conv1d(512, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn41 = nn.BatchNorm1d(512)
        self.bn42 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        
        self.conv5 = nn.Conv1d(512, 1024, 9)
        self.conv51 = nn.Conv1d(1024, 1024, 3)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn51 = nn.BatchNorm1d(1024)


        self.fc1 = nn.Linear(1024, 24)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn51(self.conv51(x)))
        #print(x.shape)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return x


# M18
class M18Net(nn.Module):
    def __init__(self):
        super(M18Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 80, 4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(64, 64, 3) 
        self.conv21 = nn.Conv1d(64, 64, 3)
        self.conv22 = nn.Conv1d(64, 64, 3)
        self.conv23 = nn.Conv1d(64, 64, 3)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn21 = nn.BatchNorm1d(64)
        self.bn22 = nn.BatchNorm1d(64)
        self.bn23 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(64, 128, 3)
        self.conv31 = nn.Conv1d(128, 128, 3)
        self.conv32 = nn.Conv1d(128, 128, 3)
        self.conv33 = nn.Conv1d(128, 128, 3)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn31 = nn.BatchNorm1d(128)
        self.bn32 = nn.BatchNorm1d(128)
        self.bn33 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)
        
        self.conv4 = nn.Conv1d(128, 256, 3)
        self.conv41 = nn.Conv1d(256, 256, 3)
        self.conv42 = nn.Conv1d(256, 256, 3)
        self.conv43 = nn.Conv1d(256, 256, 3)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn41 = nn.BatchNorm1d(256)
        self.bn42 = nn.BatchNorm1d(256)
        self.bn43 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(4)
        
        self.conv5 = nn.Conv1d(256, 512, 3)
        self.conv51 = nn.Conv1d(512, 512, 3)
        self.conv52 = nn.Conv1d(512, 512, 3)
        self.conv53 = nn.Conv1d(512, 512, 3)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn51 = nn.BatchNorm1d(512)
        self.bn52 = nn.BatchNorm1d(512)
        self.bn53 = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(512, 24)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn23(self.conv23(x)))
        x = self.pool2(x)
        
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x = self.pool3(x)
        
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn51(self.conv51(x)))
        x = F.relu(self.bn52(self.conv52(x)))
        x = F.relu(self.bn53(self.conv53(x)))
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return x

# M34-RES

def res_upsamp(A,m,n):
    upsample = nn.Upsample(size=(m,n), mode='nearest')
    A = torch.unsqueeze(A, 0)
    A = upsample(A)
    A = A.view(-1,m,n)
    return A
  
class M34Net(nn.Module):
    def __init__(self):
        super(M34Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 48, 80, 4)
        self.bn1 = nn.BatchNorm1d(48)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(48, 48, 3, padding = 1)
        self.conv21 = nn.Conv1d(48, 48, 3, padding = 1)
        self.conv22 = nn.Conv1d(48, 48, 3,padding = 1 )
        self.conv23 = nn.Conv1d(48, 48, 3,padding = 1)
        self.conv24 = nn.Conv1d(48, 48, 3,padding = 1)
        self.conv25 = nn.Conv1d(48, 48, 3,padding = 1)
        self.bn2 = nn.BatchNorm1d(48)
        self.bn21 = nn.BatchNorm1d(48)
        self.bn22 = nn.BatchNorm1d(48)
        self.bn23 = nn.BatchNorm1d(48)
        self.bn24 = nn.BatchNorm1d(48)
        self.bn25 = nn.BatchNorm1d(48)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(48, 96, 3)
        self.conv31 = nn.Conv1d(96, 96, 3)
        self.conv32 = nn.Conv1d(96, 96, 3,padding = 1)
        self.conv33 = nn.Conv1d(96, 96, 3,padding = 1)
        self.conv34 = nn.Conv1d(96, 96, 3,padding = 1)
        self.conv35 = nn.Conv1d(96, 96, 3,padding = 1)
        self.conv36 = nn.Conv1d(96, 96, 3,padding = 1)
        self.conv37 = nn.Conv1d(96, 96, 3,padding = 1)
        self.bn3 = nn.BatchNorm1d(96)
        self.bn31 = nn.BatchNorm1d(96)
        self.bn32 = nn.BatchNorm1d(96)
        self.bn33 = nn.BatchNorm1d(96)
        self.bn34 = nn.BatchNorm1d(96)
        self.bn35 = nn.BatchNorm1d(96)
        self.bn36 = nn.BatchNorm1d(96)
        self.bn37 = nn.BatchNorm1d(96)
        self.pool3 = nn.MaxPool1d(4)
        
        self.conv4 = nn.Conv1d(96, 192, 3)
        self.conv41 = nn.Conv1d(192, 192, 3)
        self.conv42 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv43 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv44 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv45 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv46 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv47 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv48 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv49 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv410 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv411 = nn.Conv1d(192, 192, 3,padding = 1)
        self.bn4 = nn.BatchNorm1d(192)
        self.bn41 = nn.BatchNorm1d(192)
        self.bn42 = nn.BatchNorm1d(192)
        self.bn43 = nn.BatchNorm1d(192)
        self.bn44 = nn.BatchNorm1d(192)
        self.bn45 = nn.BatchNorm1d(192)
        self.bn46 = nn.BatchNorm1d(192)
        self.bn47 = nn.BatchNorm1d(192)
        self.bn48 = nn.BatchNorm1d(192)
        self.bn49 = nn.BatchNorm1d(192)
        self.bn410 = nn.BatchNorm1d(192)
        self.bn411 = nn.BatchNorm1d(192)
        self.pool4 = nn.MaxPool1d(4)
        
        self.conv5 = nn.Conv1d(192, 384, 3, padding = 1)
        self.conv51 = nn.Conv1d(384, 384, 3, padding = 1)
        self.conv52 = nn.Conv1d(384, 384, 3, padding = 1)
        self.conv53 = nn.Conv1d(384, 384, 3, padding = 1)
        self.conv54 = nn.Conv1d(384, 384, 3, padding = 1)
        self.conv55 = nn.Conv1d(384, 384, 3, padding = 1)
        self.bn5 = nn.BatchNorm1d(384)
        self.bn51 = nn.BatchNorm1d(384)
        self.bn52 = nn.BatchNorm1d(384)
        self.bn53 = nn.BatchNorm1d(384)
        self.bn54 = nn.BatchNorm1d(384)
        self.bn55 = nn.BatchNorm1d(384)

        self.fc1 = nn.Linear(384, 24)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        residual = x
       
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn21(self.conv21(x)))

        x += residual 
        x = F.relu(self.bn21(x))

        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn23(self.conv23(x)))
        x += residual 
        x = F.relu(self.bn23(x))

        x = F.relu(self.bn24(self.conv24(x)))
        x = F.relu(self.bn25(self.conv25(x)))
        x += residual 
        x = F.relu(self.bn25(x))

        x = self.pool2(x)
        residual = x
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        residual = res_upsamp(residual,x.shape[1],x.shape[2])
        x += residual 
        x = F.relu(self.bn31(x))

        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x += residual 
        x = F.relu(self.bn33(x))
         
        x = F.relu(self.bn34(self.conv34(x)))
        x = F.relu(self.bn35(self.conv35(x)))
        x += residual 
        x = F.relu(self.bn35(x))

        x = F.relu(self.bn36(self.conv36(x)))
        x = F.relu(self.bn37(self.conv37(x)))
        x += residual 
        x = F.relu(self.bn37(x))
        x = self.pool3(x)
        residual = x
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        residual = res_upsamp(residual,x.shape[1],x.shape[2])
        x += residual 
        x = F.relu(self.bn41(x))

        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
        x += residual 
        x = F.relu(self.bn43(x))

        x = F.relu(self.bn44(self.conv44(x)))
        x = F.relu(self.bn45(self.conv45(x)))
        x += residual 
        x = F.relu(self.bn45(x))

        x = F.relu(self.bn46(self.conv46(x)))
        x = F.relu(self.bn47(self.conv47(x)))
        x += residual 
        x = F.relu(self.bn47(x))

        x = F.relu(self.bn48(self.conv48(x)))
        x = F.relu(self.bn49(self.conv49(x)))
        x += residual 
        x = F.relu(self.bn49(x))

        x = F.relu(self.bn410(self.conv410(x)))
        x = F.relu(self.bn411(self.conv411(x)))
        x += residual 
        x = F.relu(self.bn411(x))

      
        x = self.pool4(x)
        residual = x
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn51(self.conv51(x)))
        residual = res_upsamp(residual,x.shape[1],x.shape[2])
        x += residual 
        x = F.relu(self.bn51(x))

        x = F.relu(self.bn52(self.conv52(x)))
        x = F.relu(self.bn53(self.conv53(x)))
        x += residual 
        x = F.relu(self.bn53(x))

        x = F.relu(self.bn54(self.conv54(x)))
        x = F.relu(self.bn55(self.conv55(x)))
        x += residual 
        x = F.relu(self.bn55(x))
        
        
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return x

    
    
    
    
    
    