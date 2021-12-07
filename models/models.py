import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel = None):
        super(DownSampleBlock, self).__init__()
        if out_channel == None:
            out_channel = int(in_channel * 2)

        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, input):
        output = self.main(input)

        return output

class UpSampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel = None):
        super(UpSampleBlock, self).__init__()
        if out_channel == None:
            out_channel = int(in_channel * 0.5)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    
    def forward(self, input):
        output = self.main(input)

        return output

class Generator(nn.Module):
    def __init__(self, nc, ngf,):
        super(Generator, self).__init__()
        self.depth = 4

        self.startconv = nn.Sequential(
            nn.Conv2d(nc, ngf, 7, 1, 3, bias = False),
            nn.LeakyReLU(0.2),
        )
        
        self.DownSampleList = nn.ModuleList()
        channel = ngf
        down_channel_list = []
        for depth_num in range(self.depth ):
            downSample = DownSampleBlock(channel)
            down_channel_list.append(channel)
            channel = channel * 2
            self.DownSampleList.append(downSample)

        self.UpSampleList = nn.ModuleList()
        for depth_num in range(self.depth ):
            next_channel = int(channel * 0.5)
            upsample = UpSampleBlock(channel, next_channel)
            channel = next_channel + down_channel_list[self.depth  - depth_num - 1]
            self.UpSampleList.append(upsample)
        
        self.endconv = nn.Sequential(
            nn.Conv2d(5 * ngf, nc, 3, 1, 1, bias = False),
            nn.Tanh(),
        )

        self.down_result_list = []

    def forward(self, input):
        x = self.startconv(input)
        self.down_result_list.clear()

        for depth_num  in range(len(self.DownSampleList)):
            net = self.DownSampleList[depth_num]
            self.down_result_list.append(x)
            x = net(x)
        
        for depth_num  in range(len(self.UpSampleList)):
            net = self.UpSampleList[depth_num]
            x = net(x)
            skip = self.down_result_list[self.depth - depth_num - 1]
            x = torch.cat((x, skip), 1)

        output = self.endconv(x)

        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        depth = 5

        self.startconv = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
        )

        self.endconv = nn.Sequential(
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        channel = ndf
        self.DownSampleList = nn.ModuleList()
        for depth_num in range(depth):
            downsample = DownSampleBlock(channel)
            channel = channel * 2
                
            self.DownSampleList.append(downsample)

    def forward(self, input):
        x = self.startconv(input)

        for depth_num  in range(len(self.DownSampleList)):
            net = self.DownSampleList[depth_num]
            x = net(x)

        output = self.endconv(x)

        return output.view(-1, 1).squeeze(1)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Instance') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":

    net_G = Generator(3, 32)
    image = torch.randn(8, 3, 256, 256)
    res = net_G(image)
    print(res.shape)

    net_D = Discriminator(3, 32)
    possibility = net_D(res)
    print(possibility)