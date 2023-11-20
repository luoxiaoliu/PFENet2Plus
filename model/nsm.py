from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))

class NSM(nn.Module):
    def __init__(self, dim=3600, qdim=3600, args=None):
        super(NSM, self).__init__()

        self.conv1 = ConvBlock(dim, 256, 1)
        self.conv2 = nn.Conv2d(256, dim, 1, stride=1, padding=0)
        self.args = args

        # if args.spatial_nsm:
        #     self.spatial_conv = nn.Sequential(
        #         ConvBlock(1, 1, k=3, s=1, p=1),
        #         nn.ReLU(inplace=True),
        #         ConvBlock(1, 1, k=3, s=1, p=1),
        #         nn.ReLU(inplace=True),
        #         ConvBlock(1, 1, k=3, s=1, p=1),
        #         nn.ReLU(inplace=True),
        #     )

        # if args.spatial_encode:
        #     self.spatial_encode_conv = nn.Sequential(
        #         nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
        #         nn.BatchNorm2d(dim),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
        #         nn.BatchNorm2d(dim),
        #         nn.ReLU(inplace=True),
        #     )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, a):
        input_a = a
        # sp_size = int(a.shape[-1]**0.5)
        # if self.args.spatial_encode:
        #     tmp_a = a.clone().contiguous().view(a.shape[0], -1, sp_size, sp_size)
        #     tmp_a = tmp_a + self.spatial_encode_conv(tmp_a)
        #     a = tmp_a.contiguous().view(input_a.shape[:])

        a = a.mean(2) # b, 1, s

        # pre_sa = a.clone().contiguous().view(a.shape[0], 1, sp_size, sp_size)
        a = a.transpose(1, 2).unsqueeze(2)
        a = F.relu(self.conv1(a))
        a = self.conv2(a)

        # if self.args.spatial_nsm:
        #     if self.args.use_pre_sa:
        #         sa = self.spatial_conv(pre_sa)
        #     else:
        #         sa = self.spatial_conv(a.clone().contiguous().view(a.shape[0], 1, sp_size, sp_size))
        #     a = sa.view(a.shape[0], a.shape[1], 1, 1) + int(self.args.nsm_residue) * a

        a = a.transpose(1, 3)
        a = torch.mean(input_a * a, -1)
        b = a.size(0)
        a = a.view(b, -1)
        a = (a - a.min(1)[0].unsqueeze(1)) / (
                a.max(1)[0].unsqueeze(1) - a.min(1)[0].unsqueeze(1) + 1e-7)

        # pre_a = torch.mean(input_a, -1)
        # pre_a = pre_a.view(b, -1)
        # pre_a = (pre_a - pre_a.min(1)[0].unsqueeze(1)) / (
        #         pre_a.max(1)[0].unsqueeze(1) - pre_a.min(1)[0].unsqueeze(1) + 1e-7)  # [2,3600]

        # multiply = a * pre_a
        # print('multiply max: {}, min: {}, mean: {}'.format(multiply.max(), multiply.min(), multiply.mean())) 
        # print('pre_a max: {}, min: {}, mean: {}'.format(pre_a.max(), pre_a.min(), pre_a.mean()))   
        # print('a max: {}, min: {}, mean: {}'.format(a.max(), a.min(), a.mean()))            

        return a


class MCHNSM(nn.Module):
    def __init__(self, dim=3600, qdim=3600, args=None):
        super(MCHNSM, self).__init__()

        self.conv = nn.ModuleList([])
        for ch in args.multi_ch:
            if args.mchnsm_pre_dropout:
                self.conv.append(nn.Sequential(
                    nn.Dropout2d(p=args.mchnsm_dropout),
                    nn.Conv2d(dim, ch, 1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, dim, 1)

                ))
            else:
                self.conv.append(nn.Sequential(
                    nn.Conv2d(dim, ch, 1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=args.mchnsm_dropout),
                    nn.Conv2d(ch, dim, 1)

                ))
        self.args = args

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, a):
        input_a = a
        sp_size = int(a.shape[-1] ** 0.5)

        a = a.mean(2)  # [2, 1, 3600]   # b, 1, s
        pre_sa = a.clone().contiguous().view(a.shape[0], 1, sp_size, sp_size)
        a = a.transpose(1, 2).unsqueeze(2)

        a_list = []
        for c_idx, ch in enumerate(self.args.multi_ch):
            tmp_a = self.conv[c_idx](a)
            a_list.append(tmp_a)
        a = sum(a_list) / len(a_list)

        a = a.transpose(1, 3)
        a = torch.mean(input_a * a, -1)
        b = a.size(0)
        a = a.view(b, -1)
        a = (a - a.min(1)[0].unsqueeze(1)) / (
                a.max(1)[0].unsqueeze(1) - a.min(1)[0].unsqueeze(1) + 1e-7)

        pre_a = torch.mean(input_a, -1)
        pre_a = pre_a.view(b, -1)
        pre_a = (pre_a - pre_a.min(1)[0].unsqueeze(1)) / (
                pre_a.max(1)[0].unsqueeze(1) - pre_a.min(1)[0].unsqueeze(1) + 1e-7)

        return a


class MSNSM(nn.Module):
    def __init__(self, dim=3600, args=None):
        super(MSNSM, self).__init__()

        self.args = args
        self.bins = args.msnsm_bins
        self.conv1 = nn.ModuleList([])
        self.conv2 = nn.ModuleList([])
        for bin in self.bins:
            self.conv1.append(nn.Sequential(
                nn.Conv2d(dim // (bin ** 2), 256 // bin, 1),
                nn.BatchNorm2d(256 // bin),
                nn.ReLU(inplace=True)
            ))
            self.conv2.append(
                nn.Conv2d(256 // bin, dim // (bin ** 2), 1)
            )

        if self.args.msnsm_type == 'conv':
            self.merge = nn.Linear(len(self.bins), 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, a):  # [b, 1, q, s]
        input_a = a
        sp_size = int(a.shape[-1] ** 0.5)
        bs = a.shape[0]
        a = a.mean(2)   # b, 1, s
        pre_sa = a.clone().contiguous().view(a.shape[0], 1, sp_size, sp_size)

        a_list = []
        for idx, bin in enumerate(self.bins):
            rep_a = pre_sa.contiguous().view(bs, 1, bin, sp_size // bin, bin, sp_size // bin)  # b, 1, ph, h, pw, w
            rep_a = rep_a.permute(0, 1, 2, 4, 3, 5)  # [b, 1, p, h, p, w] -> [b, 1, ph, pw, h, w]
            rep_a = rep_a.contiguous().view(bs, 1, bin * bin,
                                            sp_size // bin * sp_size // bin)  # [b, 1, p, p, h, w] -> [b, 1, pp, hw]
            rep_a = rep_a.permute(0, 3, 1, 2)  # [b, 1, pp, hw] -> [b, hw, 1, pp]
            rep_a = self.conv1[idx](rep_a)
            rep_a = self.conv2[idx](rep_a)  # [b, hw, 1, pp]
            rep_a = rep_a.permute(0, 2, 3, 1)  # [b, hw, 1, pp] -> [b, 1, pp, hw]
            rep_a = rep_a.contiguous().view(bs, 1, bin, bin, sp_size // bin, sp_size // bin)  # [b, 1, ph, pw, h, w]
            rep_a = rep_a.permute(0, 1, 2, 4, 3, 5)  # [b, 1, ph, pw, h, w] -> [b, 1, ph, h, pw, w]
            rep_a = rep_a.contiguous().view(bs, 1, sp_size, sp_size)
            rep_a = rep_a.contiguous().view(bs, 1, 1, sp_size * sp_size)  # b, 1, 1, 3600
            tmp_a = torch.mean(input_a * rep_a, -1)  # b, 1, 3600
            a_list.append(tmp_a)

        if self.args.msnsm_type == 'mean':
            a = sum(a_list) / len(a_list)
        elif self.args.msnsm_type == 'max':
            a = torch.cat(a_list, 1).max(1)[0].unsqueeze(1)
        elif self.args.msnsm_type == 'conv':
            a = torch.cat(a_list, 1).permute(0, 2, 1)
            a = self.merge(a).permute(0, 2, 1)

        b = a.size(0)
        a = a.view(b, -1)
        a = (a - a.min(1)[0].unsqueeze(1)) / (
                a.max(1)[0].unsqueeze(1) - a.min(1)[0].unsqueeze(1) + 1e-7)

        pre_a = torch.mean(input_a, -1)
        pre_a = pre_a.view(b, -1)
        pre_a = (pre_a - pre_a.min(1)[0].unsqueeze(1)) / (
                pre_a.max(1)[0].unsqueeze(1) - pre_a.min(1)[0].unsqueeze(1) + 1e-7)

        return a
