import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Baseline Model (which is just taking the black and white dataset verbatim)
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()

    
    def forward(self, x):
        return x

#Define A Double Convolution (As used in the U-Net architecture)
def double_conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   

#Define the Generator of the GAN, which also serves standalone as the baseline
#Fully-Convolutional Network based on the U-Net Architechure
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #3 x 64 x 64 image
        self.conv_encode1 = double_conv(3, 64, 2)
        #64 x 32 x 32
        self.conv_encode2 = double_conv(64, 128, 2)
        #128 x 16 x 16
        self.conv_encode3 = double_conv(128, 256, 2)
        #256 x 8 x 8

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        
        #256 x 16 x 16
        self.conv_decode1 = double_conv(256 + 128, 128)
        #128 x 32 x 32
        self.conv_decode2 = double_conv(128 + 64, 64)
        #64 x 64 x 64
        
        self.output = nn.Conv2d(64, 3, 1)
        #3 x 64 x 64

    
    def forward(self, x):
        conv1 = self.conv_encode1(x)
        conv2 = self.conv_encode2(conv1)
        x = self.conv_encode3(conv2)
        
        x = self.upsample(x) 
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_decode1(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_decode2(x)
        
        x = self.upsample(x)
        x = self.output(x)
        
        return x
    
#Define the Discriminator of the GAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #3 x 64 x 64 image
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias= False),
            nn.LeakyReLU(0.1, inplace = True), 
            nn.Dropout(0.2))
        #64 x 32 x 32
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias= False),
            nn.LeakyReLU(0.1, inplace = True), 
            nn.BatchNorm2d(128),
            nn.Dropout(0.2))
        #128 x 16 x 16
        self.convLayer3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias= False),
            nn.LeakyReLU(0.1, inplace = True), 
            nn.BatchNorm2d(256),
            nn.Dropout(0.3))
        #256 x 8 x 8

        self.LinLayer1 = nn.Linear(16384, 1000)
        self.LinLayer2 = nn.Linear(1000, 100)
        self.FinalLayer = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        x = self.convLayer3(x)
        x = x.view(-1, 16384)
        x = F.relu(self.LinLayer1(x))
        x = F.relu(self.LinLayer2(x))
        x = self.FinalLayer(x)
        return x
#Initialize baseline model
def initialize_baseline():
    model = Baseline()
    loss_fnc = nn.MSELoss()

    return model, loss_fnc

#Initialize U-Net as standalone model
def initialize_model(lr):
    model = Generator()
    loss_fnc = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) #beta value not set 

    return model, loss_fnc, optimizer

#Initialize U-Net as generator
def initialize_model_G(lr):
    model = Generator()
    loss_fnc = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) #beta value not set 

    return model, loss_fnc, optimizer

#Initialize Discrimnator
def initialize_model_D(lr):
    model = Discriminator()
    loss_fnc = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) #beta value not set 

    return model, loss_fnc, optimizer


