import torch
import torch.nn as nn
import torch.functional as F



class DepthBranch(nn.Module):
    def __init__(self):
        super(DepthBranch, self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, stride=1), nn.ReLU(True)
        )
        self.conv21 = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1, stride=1), nn.ReLU(True)
        )
        self.conv31 = nn.Sequential(
            nn.Conv2d(24, 64, 3, padding=1, stride=1), nn.ReLU(True)
        )
        self.conv41 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, stride=1), nn.ReLU(True)
        )
        self.conv51 = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1, stride=1), nn.ReLU(True)
        )




        self.convpool1 = nn.Sequential(
            nn.Conv2d(64, 16, 3, groups=16, padding=1, stride=2), nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),
        )
        self.convpool2 = nn.Sequential(
            nn.Conv2d(64, 24, 3, groups=8, padding=1, stride=2), nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),
        )
        self.convpool3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, groups=32, padding=1, stride=2), nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),
        )
        self.convpool4 = nn.Sequential(
            nn.Conv2d(64, 96, 3, groups=32, padding=1, stride=2), nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),
        )
        self.convpool5 = nn.Sequential(
            nn.Conv2d(64, 320, 3, groups=64, padding=1, stride=2), nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),
        )


    def forward(self, d):
        d11 = self.conv11(d) 
        d1 = self.convpool1(d11)
        d21 = self.conv21(d1) 
        d2 = self.convpool2(d21)
        d31 = self.conv31(d2) 
        d3 = self.convpool3(d31)
        d41 = self.conv41(d3) 
        d4 = self.convpool4(d41)
        d51 = self.conv51(d4) 
        d5 = self.convpool5(d51)

        return d1, d2, d3, d4, d5

if __name__=='__main__':
    import torch
    depthnet = DepthNet()
    depth = torch.randn(1, 3, 352, 352)
    out = depthnet(depth)
    for i in out:
        print(i.shape)