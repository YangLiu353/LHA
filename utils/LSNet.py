import torch
import torch.nn as nn
# import torch.functional as F
import torch.nn.functional as F

class AFD_semantic(nn.Module):
    '''
    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

    def __init__(self, in_channels, att_f):
        super(AFD_semantic, self).__init__()
        mid_channels = int(in_channels * att_f)

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, 1, 1, bias=True)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):

        fm_t_pooled = self.avg_pool(fm_t)
        rho = self.attention(fm_t_pooled)
        rho = torch.sigmoid(rho.squeeze())
        rho = rho / torch.sum(rho, dim=1, keepdim=True)

        fm_s_norm = torch.norm(fm_s, dim=(2, 3), keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=(2, 3), keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)

        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=(2, 3))
        loss = loss.sum(1).mean(0)

        return loss


class AFD_spatial(nn.Module):
    '''
    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

    def __init__(self, in_channels):
        super(AFD_spatial, self).__init__()

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, 1, 3, 1, 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):

        rho = self.attention(fm_t)
        rho = torch.sigmoid(rho)
        rho = rho / torch.sum(rho, dim=(2,3), keepdim=True)

        fm_s_norm = torch.norm(fm_s, dim=1, keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=1, keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)
        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=1, keepdim=True)
        loss =torch.sum(loss,dim=(2,3)).mean(0)
        return loss



class DSConvolution(nn.Module):
    # Use depthwise separable convolution optimization
    def __init__(self, in_channels, out_channels):
        super(DSConvolution, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu6(x, inplace=True)  # Use ReLU6 to activate the function
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu6(x, inplace=True)  # Use ReLU6 to activate the function
        return x

import torch.nn.functional as F

class SEM(nn.Module):
    def __init__(self, in_channels):
        super(SEM, self).__init__()
        self.reduced_channels = in_channels // 2

        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, 3, padding=1, dilation=1, groups=self.reduced_channels),
            nn.BatchNorm2d(self.reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduced_channels, in_channels, 1))

        self.spatial_attention2 = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, 3, padding=3, dilation=3, groups=self.reduced_channels),
            nn.BatchNorm2d(self.reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduced_channels, in_channels, 1))

        self.spatial_attention3 = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, 3, padding=5, dilation=5, groups=self.reduced_channels),
            nn.BatchNorm2d(self.reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduced_channels, in_channels, 1))

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_r):
        SA1 = self.spatial_attention1(x_r)
        SA2 = self.spatial_attention2(x_r)
        SA3 = self.spatial_attention3(x_r)

        SA = SA1 + SA2 + SA3
        SA = self.gamma * SA + x_r

        return SA

class MAModule(nn.Module):
    def __init__(self, channels, reduction_ratio = 16 ):
        super(MAModule, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Channel attention function
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Self-attention function
        self.self_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Axial attention function
        self.axial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Use lightweight convolution modules
        self.conv_module = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        module_input = x

        # Self-attention
        x_self = self.self_attention(x) * x

        # Channel attention
        x_ch = self.global_pool(x)
        x_ch = self.conv_module(x_ch) * x

        # Axial attention

        x_axial = self.axial_attention(x)
        x_axial = x_axial.mean(dim=1, keepdim=True)

        # Combined attention
        x = x_ch + x_self
        x = x * x_axial
        return module_input * x

class CrossChannelFusion(nn.Module):
    def __init__(self, in_channels):
        super(CrossChannelFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels*2, in_channels, 1) # Cut the number of channels in half
        self.mobilenet = DSConvolution(in_channels, 160)
        self.sem = SEM(in_channels=160)
        self.mam = MAModule(160)

    def forward(self, x1, x2):
        out = torch.cat((x1, x2), dim=1)  # Before adding a module with half the number of channels, you need to concatenate two input tensors along the channel dimension
        out = self.conv(out)  # Cut the number of channels in half
        out = self.mobilenet(out)
        out = self.sem(out)
        out = self.mam(out)
        return out



# The following is the - boundary module
class PyramidConv(nn.Module):
    def __init__(self, channel, scales=4):
        super(PyramidConv, self).__init__()
        self.width = channel // scales
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.width, self.width, 3, stride=1, padding=1, groups=40, dilation=1, bias=False), nn.BatchNorm2d(self.width), nn.PReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.width, self.width, 3, stride=1, padding=2, groups=40, dilation=2, bias=False), nn.BatchNorm2d(self.width), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.width, self.width, 3, stride=1, padding=4, groups=40, dilation=4, bias=False), nn.BatchNorm2d(self.width), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.width, self.width, 3, stride=1, padding=6, groups=40, dilation=6, bias=False), nn.BatchNorm2d(self.width), nn.PReLU()
        )

    def forward(self, x):
        sx = torch.split(x, self.width, 1)

        sp = self.conv1(sx[0])
        x  = sp

        sp = self.conv2(sx[1] + sp)
        x  = torch.cat((x, sp), 1)

        sp = self.conv3(sx[2] + sp)
        x  = torch.cat((x, sp), 1)

        sp = self.conv4(sx[3] + sp)
        x  = torch.cat((x, sp), 1)

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc(avg_out)
        out = avg_out
        return out

class AlternateGuidance(nn.Module):
    def __init__(self, in_channel, channel, echannel, egroups):
        super(AlternateGuidance, self).__init__()

        self.converta = nn.Sequential(
            nn.Conv2d(in_channel, channel, 1, groups=channel, bias=False), nn.BatchNorm2d(channel), nn.PReLU(),
        )
        self.convertb = nn.Sequential(
            nn.Conv2d(in_channel, channel, 1, groups=channel, bias=False), nn.BatchNorm2d(channel), nn.PReLU(),
        )

        self.ca1 = ChannelAttention(channel)
        self.ca2 = ChannelAttention(channel)
        
        if in_channel == 160:
            self.convs1a = PyramidConv(channel)
            self.convs1b = PyramidConv(channel)
        else:
            self.convs1a = nn.Sequential(
            nn.Conv2d(channel+channel, channel, 3, stride=1, padding=1, groups=channel, bias=False), nn.BatchNorm2d(channel), nn.PReLU()
        )

            self.convs1b = nn.Sequential(
            nn.Conv2d(channel+channel, channel, 3, stride=1, padding=1, groups=channel, bias=False), nn.BatchNorm2d(channel), nn.PReLU()
        )

        self.convs2a = nn.Sequential(
            nn.Conv2d(channel+channel, channel, 3, stride=1, padding=1, groups=channel, bias=False), nn.BatchNorm2d(channel), nn.PReLU()
        )

        self.convs2b = nn.Sequential(
            nn.Conv2d(channel+channel, channel, 3, stride=1, padding=1, groups=channel, bias=False), nn.BatchNorm2d(channel), nn.PReLU()
        )

        self.convs3a = nn.Sequential(
            nn.Conv2d(channel+channel, channel, 3, stride=1, padding=1, groups=channel, bias=False), nn.BatchNorm2d(channel), nn.PReLU()
        )

        self.convs3b = nn.Sequential(
            nn.Conv2d(channel+channel, channel, 3, stride=1, padding=1, groups=channel, bias=False), nn.BatchNorm2d(channel), nn.PReLU()
            )

        self.scorea = nn.Conv2d(channel, channel, 3, groups=channel, padding=1)
        self.scoreb = nn.Conv2d(channel, channel, 3, groups=channel, padding=1)
        self.conve = nn.Conv2d(echannel, channel, 3, groups=egroups, padding=1)
        self.convy = nn.Conv2d(echannel, channel, 3, groups=egroups, padding=1)

    def forward(self, x, y, e):
        xa = self.converta(x)
        xb = self.convertb(x)

        xa = xa * self.ca1(xa)
        xb = xb * self.ca2(xb)

        if e is not None:
            e = self.conve(e)
            xa = torch.cat((xa, e), 1)
        if y is not None:
            y = self.convy(y)
            xb = torch.cat((xb, y), 1)
        xa = self.convs1a(xa)
        xb = self.convs1b(xb)
        xa = self.convs2a(torch.cat((xa, xb), 1))
        xb = self.convs2b(torch.cat((xa, xb), 1))
        xa = self.convs3a(torch.cat((xa, xb), 1))
        xb = self.convs3b(torch.cat((xa, xb), 1))
        if e is None:
            e = self.scorea(xa)
        else:
            e = self.scorea(xa) + e
        if y is None:
            y = self.scoreb(xb)
        else:
            y = self.scoreb(xb) + y

        return y, e
# Above is the - boundary module

from utils.mobilenetv2 import mobilenet_v2
from utils.DepthBranch import DepthBranch
class LSNet(nn.Module):
    def __init__(self):
        super(LSNet, self).__init__()
        # rgb,depth encode
        self.rgb_pretrained = mobilenet_v2()
        # self.depth_pretrained = mobilenet_v2()
        self.depth_pretrained = DepthBranch()
        # Upsample_model
        self.upsample1_g = nn.Sequential(nn.Conv2d(212, 34, 3, 1, 1, ), nn.BatchNorm2d(34), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample2_g = nn.Sequential(nn.Conv2d(240, 52, 3, 1, 1, ), nn.BatchNorm2d(52), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample3_g = nn.Sequential(nn.Conv2d(288, 80, 3, 1, 1, ), nn.BatchNorm2d(80), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample4_g = nn.Sequential(nn.Conv2d(320, 128, 3, 1, 1, ), nn.BatchNorm2d(128), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample5_g = nn.Sequential(nn.Conv2d(160, 160, 3, 1, 1, ), nn.BatchNorm2d(160), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))
        # Cross-channel fusion
        self.cross_channel_fusion5 = CrossChannelFusion(320)
        self.cross_channel_fusion4 = CrossChannelFusion(96)
        self.cross_channel_fusion3 = CrossChannelFusion(32)
        self.cross_channel_fusion2 = CrossChannelFusion(24)
        self.cross_channel_fusion1 = CrossChannelFusion(16)


        self.UPsample1_g = nn.UpsamplingBilinear2d(scale_factor=2, )
        self.UPsample2_g = nn.UpsamplingBilinear2d(scale_factor=2, )
        self.UPsample3_g = nn.UpsamplingBilinear2d(scale_factor=2, )
        self.UPsample4_g = nn.UpsamplingBilinear2d(scale_factor=2, )
        self.UPsample5_g = nn.UpsamplingBilinear2d(scale_factor=2, )


        self.conv1_g = nn.Conv2d(34, 1, 1)
        self.conv2_g = nn.Conv2d(52, 1, 1)
        self.conv3_g = nn.Conv2d(80, 1, 1)
        self.conv4_g = nn.Conv2d(128, 1, 1)
        self.conv5_g = nn.Conv2d(160, 1, 1)


        # Add border
        self.agd1 = AlternateGuidance(34, 34, 52, 2)
        self.agd2 = AlternateGuidance(52, 52, 80, 4)
        self.agd3 = AlternateGuidance(80, 80, 128, 16)
        self.agd4 = AlternateGuidance(128, 128, 160, 32)
        self.agd5 = AlternateGuidance(160, 160, 160, 160)


        # Tips: speed test and params and more this part is not included.
        # please comment this part when involved.
        if self.training:
            self.AFD_semantic_5_R_T = AFD_semantic(320,0.0625)
            self.AFD_semantic_4_R_T = AFD_semantic(96,0.0625)
            self.AFD_semantic_3_R_T = AFD_semantic(32,0.0625)
            self.AFD_spatial_3_R_T = AFD_spatial(32)
            self.AFD_spatial_2_R_T = AFD_spatial(24)
            self.AFD_spatial_1_R_T = AFD_spatial(16)


    def forward(self, rgb, ti):
        # rgb
        A1, A2, A3, A4, A5 = self.rgb_pretrained(rgb)
        # ti
        A1_t, A2_t, A3_t, A4_t, A5_t = self.depth_pretrained(ti)

        A1_size = rgb.size()[2:]
        A2_size = A1.size()[2:]
        A3_size = A2.size()[2:]
        A4_size = A3.size()[2:]
        A5_size = A4.size()[2:]

        F5 = self.cross_channel_fusion5(A5_t, A5)
        F4 = self.cross_channel_fusion4(A4_t, A4)
        F3 = self.cross_channel_fusion3(A3_t, A3)
        F2 = self.cross_channel_fusion2(A2_t, A2)
        F1 = self.cross_channel_fusion1(A1_t, A1)


        F5_ = self.upsample5_g(F5)
        F5, e5 = self.agd5(F5_, None, None)
        F4 = torch.cat((F4, F5_), dim=1)
        F4_ = self.upsample4_g(F4)
        F5_4 = self.UPsample1_g(F5)
        e5_4 = self.UPsample1_g(e5)
        F4, e4 = self.agd4(F4_, F5_4, e5_4)
        F3 = torch.cat((F3, F4_), dim=1)
        F3_ = self.upsample3_g(F3)
        F4_3 = self.UPsample1_g(F4)
        e4_3 = self.UPsample1_g(e4)
        F3, e3 = self.agd3(F3_, F4_3, e4_3)
        F2 = torch.cat((F2, F3_), dim=1)
        F2_ = self.upsample2_g(F2)
        F3_2 = self.UPsample1_g(F3)
        e3_2 = self.UPsample1_g(e3)
        F2, e2 = self.agd2(F2_, F3_2, e3_2)
        F1 = torch.cat((F1, F2_), dim=1)
        F1_ = self.upsample1_g(F1)
        F2_1 = self.UPsample1_g(F2)
        e2_1 = self.UPsample1_g(e2)
        F1, e1 = self.agd1(F1_, F2_1, e2_1)

        F5 = self.conv5_g(F5)
        F5 = F.interpolate(F5, A5_size, mode='bilinear', align_corners=True)
        F4 = self.conv4_g(F4)
        F4 = F.interpolate(F4, A4_size, mode='bilinear', align_corners=True)
        F3 = self.conv3_g(F3)
        F3 = F.interpolate(F3, A3_size, mode='bilinear', align_corners=True)
        F2 = self.conv2_g(F2)
        F2 = F.interpolate(F2, A2_size, mode='bilinear', align_corners=True)
        F1 = self.conv1_g(F1)
        F1 = F.interpolate(F1, A1_size, mode='bilinear', align_corners=True)

        e5 = self.conv5_g(e5)
        e5 = F.interpolate(e5, A5_size, mode='bilinear', align_corners=True)
        e4 = self.conv4_g(e4)
        e4 = F.interpolate(e4, A4_size, mode='bilinear', align_corners=True)
        e3 = self.conv3_g(e3)
        e3 = F.interpolate(e3, A3_size, mode='bilinear', align_corners=True)
        e2 = self.conv2_g(e2)
        e2 = F.interpolate(e2, A2_size, mode='bilinear', align_corners=True)
        e1 = self.conv1_g(e1)
        e1 = F.interpolate(e1, A1_size, mode='bilinear', align_corners=True)



        if self.training:
            loss_semantic_5_R_T = self.AFD_semantic_5_R_T(A5, A5_t.detach())
            loss_semantic_5_T_R = self.AFD_semantic_5_R_T(A5_t, A5.detach())
            loss_semantic_4_R_T = self.AFD_semantic_4_R_T(A4, A4_t.detach())
            loss_semantic_4_T_R = self.AFD_semantic_4_R_T(A4_t, A4.detach())
            loss_semantic_3_R_T = self.AFD_semantic_3_R_T(A3, A3_t.detach())
            loss_semantic_3_T_R = self.AFD_semantic_3_R_T(A3_t, A3.detach())
            loss_spatial_3_R_T = self.AFD_spatial_3_R_T(A3, A3_t.detach())
            loss_spatial_3_T_R = self.AFD_spatial_3_R_T(A3_t, A3.detach())
            loss_spatial_2_R_T = self.AFD_spatial_2_R_T(A2, A2_t.detach())
            loss_spatial_2_T_R = self.AFD_spatial_2_R_T(A2_t, A2.detach())
            loss_spatial_1_R_T = self.AFD_spatial_1_R_T(A1, A1_t.detach())
            loss_spatial_1_T_R = self.AFD_spatial_1_R_T(A1_t, A1.detach())
            loss_KD = loss_semantic_5_R_T + loss_semantic_5_T_R + \
                      loss_semantic_4_R_T + loss_semantic_4_T_R + \
                      loss_semantic_3_R_T + loss_semantic_3_T_R + \
                      loss_spatial_3_R_T + loss_spatial_3_T_R + \
                      loss_spatial_2_R_T + loss_spatial_2_T_R + \
                      loss_spatial_1_R_T + loss_spatial_1_T_R

            return F1, F2, F3, F4, F5, loss_KD, e1, e2, e3, e4, e5
        return F1, e1
