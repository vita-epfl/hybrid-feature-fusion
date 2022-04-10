import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from .basenet import *


class Res18Crop(torch.nn.Module):
    """
    CNN for single frame model with cropped image.
    """

    def __init__(self, backbone, drop_p=0.5):
        super(Res18Crop, self).__init__()
        self.backbone = backbone
        self.conv1 = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_p)
        self.act = nn.Sigmoid()
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.view(-1, 1024)
        x = self.linear(x)
        x = self.act(x)

        return x


class Res18RoI(torch.nn.Module):
    """
    CNN for single frame model. ResNet-18 bacbone and RoI for adding context.
    """

    def __init__(self, resnet, last_block, drop_p=0.5):
        super(Res18RoI, self).__init__()
        self.resnet = resnet
        self.last_block = last_block.apply(set_conv2d_stride1)
        self.conv_last = torch.nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_p)
        self.FC = torch.nn.Linear(1024, 1)
        self.act = torch.nn.Sigmoid()

    def forward(self, imgs, bboxes):
        feature_maps = self.resnet(imgs)
        fa = RoIAlign(output_size=(7, 7), spatial_scale=1 / 8,
                      sampling_ratio=2, aligned=True)
        ya = fa(feature_maps, bboxes)
        y = self.last_block(ya)
        y = self.conv_last(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = y.view(1, -1)
        y = self.FC(y)
        y = self.act(y)

        return y


class DecoderRNN_PV(nn.Module):
    def __init__(self, h_RNN_layers=1, h_RNN=32,
                 h_FC_dim=16, drop_p=0.0):
        super(DecoderRNN_PV, self).__init__()

        self.h_RNN = h_RNN# RNN hidden nodes
        self.h_FC_dim = h_FC_dim
    
        # position-velocity decoder
        self.RNN = nn.LSTM(
            input_size=8,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       #  (batch, time_step, input_size)
        )
        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.dropout = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(self.h_FC_dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, x_pv, x_lengths):        
        # N, T, n = x_3d.size()
        # use input of descending length
        packed_x1_RNN = torch.nn.utils.rnn.pack_padded_sequence(x_pv, x_lengths, 
                                                                batch_first=True, enforce_sorted=False)
        self.RNN.flatten_parameters()
        
        packed_RNN_out_1, _ = self.RNN(packed_x1_RNN, None)
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        RNN_out_1, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out_1, batch_first=True)
        RNN_out_1 = RNN_out_1.contiguous()
        # choose RNN_out at the last time step
        output_1 = RNN_out_1[:, -1, :]
        xpv = self.fc1(output_1)
        xpv = F.relu(xpv)
        xpv = self.dropout(xpv)
        x = xpv
        x = self.fc2(x)
        x = self.act(x)

        return x