import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import numpy as np
from Util.util import Interpolate, UnNormfunc

# UNet, ResNet, Discriminator, NestUNet
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, residual=0):
        super(UNet, self).__init__() 


        nb_filter = [64, 128, 256, 512, 1024]

        self.conv_enc_0_1 = nn.Conv2d(in_channels, nb_filter[0], kernel_size=3, padding=1)
        self.bn_enc_0_1 = nn.BatchNorm2d(nb_filter[0])
        self.relu_enc_0_1 = nn.ReLU()

        self.enc_block0 = nn.Sequential(
                nn.Conv2d(in_channels, nb_filter[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[0]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[0]),
                nn.ReLU()
            )

        self.pool0 = nn.MaxPool2d(2, 2)

        self.enc_block1 = nn.Sequential(
                nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[1]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[1], nb_filter[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[1]),
                nn.ReLU(),
            )

        self.pool1 = nn.MaxPool2d(2, 2)


        self.enc_block2 = nn.Sequential(
                nn.Conv2d(nb_filter[1], nb_filter[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[2]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[2], nb_filter[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[2]),
                nn.ReLU(),
            )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc_block3 = nn.Sequential(
                nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[3]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[3], nb_filter[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[3]),
                nn.ReLU(),
            )

        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc_block4 = nn.Sequential(
                nn.Conv2d(nb_filter[3], nb_filter[4], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[4]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[4], nb_filter[4], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[4]),
                nn.ReLU(),
            )

        self.up4 = Interpolate(scale_factor=2, mode='bilinear')

        self.dec_block3 = nn.Sequential(
                nn.Conv2d(nb_filter[4]+nb_filter[3], nb_filter[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[3]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[3], nb_filter[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[3]),
                nn.ReLU(),
            )

        self.up3 = Interpolate(scale_factor=2, mode='bilinear')

        self.dec_block2 = nn.Sequential(
                nn.Conv2d(nb_filter[3]+nb_filter[2], nb_filter[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[2]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[2], nb_filter[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[2]),
                nn.ReLU(),
            )

        self.up2 = Interpolate(scale_factor=2, mode='bilinear')


        self.dec_block1 = nn.Sequential(
                nn.Conv2d(nb_filter[2]+nb_filter[1], nb_filter[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[1]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[1], nb_filter[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[1]),
                nn.ReLU(),
            )

        self.up3 = Interpolate(scale_factor=2, mode='bilinear')

        self.dec_block0 = nn.Sequential(
                nn.Conv2d(nb_filter[1]+nb_filter[0], nb_filter[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[0]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[0]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[0], out_channels, kernel_size=3, padding=1),
            )




    def forward(self, x):
        
        x0_0 = self.enc_block0(x)
        x0_1 = self.pool0(x0_0)
        x1_0 = self.enc_block1(x0_1)
        x1_1 = self.pool1(x1_0)
        x2_0 = self.enc_block2(x1_1)
        x2_1 = self.pool2(x2_0)
        x3_0 = self.enc_block3(x2_1)
        x3_1 = self.pool3(x3_0)
        x4_0 = self.enc_block4(x3_1)
        x4_2 = torch.cat([self.up4(x4_0), x3_0], 1)
        x3_3 = self.dec_block3(x4_2)
        x3_2 = torch.cat([self.up4(x3_3), x2_0], 1)
        x2_3 = self.dec_block2(x3_2)
        x2_2 = torch.cat([self.up4(x2_3), x1_0], 1)
        x1_3 = self.dec_block1(x2_2)
        x1_2 = torch.cat([self.up4(x1_3), x0_0], 1)
        x0_3 = self.dec_block0(x1_2)

        #x_out = x * x0_3
        if residual:
            output = x - x0_3
        else:
            output = x0_3

        return output
        
class ResNet(nn.Module):
    def __init__(self, channels=3,num_of_layers=17,features=64):
        super(ResNet, self).__init__() 
        kernel_size = 3
        padding = 1
        layers = []
        
        # 1st layer
        conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        nn.init.xavier_normal_(conv1.weight)
        layers.append(conv1)
        layers.append(nn.ReLU(inplace=True))

        # 2nd~L-1th layer
        for _ in range(num_of_layers-2):
            conv_mid = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
            nn.init.xavier_normal_(conv_mid.weight)
            layers.append(conv_mid)
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        conv_last = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=padding, bias=False)
        nn.init.xavier_normal_(conv_last.weight)
        layers.append(conv_last)
        self.resderainnet = nn.Sequential(*layers)

    def forward(self, x):
        output = x - self.resderainnet(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__() 
        
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.LeakyReLU(0.2, inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


class NestedUNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(args.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], args.input_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], args.input_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], args.input_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], args.input_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], args.input_channels, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
