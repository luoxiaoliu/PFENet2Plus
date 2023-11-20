import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models


from torch.autograd import Variable
from model.nsm import NSM, MSNSM, MCHNSM
import math
import os


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class PFENet(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d,
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False, args=None):
        super(PFENet, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales
        self.vgg = vgg

        models.BatchNorm = BatchNorm

        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
            self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512

        if args.use_mid_nsm:
            self.theta_mid = nn.Sequential(
                nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5)
            )

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        if args.strided_theta:
            self.theta = nn.Sequential(
                nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=False, groups=2048),
                # nn.ReLU(inplace=True),
                nn.Conv2d(2048, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5)
            )
        else:
            self.theta = nn.Sequential(
                nn.Conv2d(2048 + int(args.concat_mid) * (1024 + 512), reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5)
            )

        if len(args.msnsm_bins) > 0:
            nsm_module = MSNSM
        elif len(args.multi_ch) > 0:
            nsm_module = MCHNSM
        else:
            nsm_module = NSM
        if args.train_h == 473:
            init_dim = 3600
        elif args.train_h == 417:
            init_dim = 2704

        if args.down_size > 0:
            if args.multi_nsm:
                self.nsm = nn.ModuleList([])
                for i in range(3):
                    self.nsm.append(nsm_module(dim=init_dim // (args.down_size ** 2), args=args))
            else:
                self.nsm = nsm_module(dim=init_dim // (args.down_size ** 2), args=args)

            if args.use_mid_nsm:
                self.mid_nsm = nsm_module(dim=init_dim // (args.down_size ** 2), args=args)
        elif args.strided_theta:
            if args.multi_nsm:
                self.nsm = nn.ModuleList([])
                for i in range(3):
                    self.nsm.append(nsm_module(dim=900, args=args))
            else:
                self.nsm = nsm_module(dim=900, args=args)

            if args.use_mid_nsm:
                self.mid_nsm = nsm_module(dim=900, args=args)
        else:
            if args.multi_nsm:
                self.nsm = nn.ModuleList([])
                for i in range(3):
                    self.nsm.append(nsm_module(dim=3600, args=args))
            else:
                self.nsm = nsm_module(dim=3600, args=args)

            if args.use_mid_nsm:
                self.mid_nsm = nsm_module(dim=3600, args=args)

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        factor = 1
        mask_add_num = len(args.multi_scales) + int(args.use_mid_nsm) * len(args.mid_multi_scales)

        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            if args.real_multi_supp:
                self.init_merge.append(nn.Sequential(
                    nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                ))
            else:
                self.init_merge.append(nn.Sequential(
                    nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))


        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins) - 1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

        self.args = args

    def forward(self, x, s_x=torch.FloatTensor(1, 1, 3, 417, 417).cuda(), s_y=torch.FloatTensor(1, 1, 417, 417).cuda(),
                y=None):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)

        query_feat1 = torch.cat([query_feat_3, query_feat_2], 1)
        full_query_feat = query_feat1.clone()
        query_feat = self.down_query(query_feat1)

        #   Support Feature
        supp_feat_list = []

        final_supp_list = []
        full_supp_list = []
        mask_list = []
        cosine_eps = 1e-7

        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                mask_list.append(mask)
                supp_feat_4 = self.layer4(supp_feat_3 * mask)
                if self.args.concat_mid:
                    cat_mid_supp_feat = torch.cat([supp_feat_2, supp_feat_3, supp_feat_4], 1) * mask
                    final_supp_list.append(cat_mid_supp_feat)
                else:
                    final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)

            supp_feat1 = torch.cat([supp_feat_3, supp_feat_2], 1)
            full_supp_list.append(supp_feat1)
            supp_feat2 = self.down_supp(supp_feat1)

            supp_feat = Weighted_GAP(supp_feat2, mask)
            supp_feat_list.append(supp_feat)

        supp_feat = supp_feat_list[0]
        if self.shot > 1:
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]

            supp_feat /= len(supp_feat_list)

        #### nsm for high feature
        if self.args.concat_mid:
            q = self.theta(torch.cat([query_feat_2, query_feat_3, query_feat_4], 1))
        else:
            q = self.theta(query_feat_4)
        q = F.normalize(q, 2, 1)
        multi_scales = self.args.multi_scales
        paddings = self.args.paddings
        multi_scales_mask_list = [[], [], []]
        response_mask_list = []
        if self.args.query_down_size > 1:
            q = F.interpolate(q,
                              size=(q.shape[-2] // self.args.query_down_size, q.shape[-1] // self.args.query_down_size),
                              mode='bilinear', align_corners=True)
        for j in range(len(multi_scales)):
            cosine_eps = 1e-7
            for i, tmp_supp_feat in enumerate(final_supp_list):
                b0, c0, w0, _ = tmp_supp_feat.size()
                tmp_mask = F.interpolate(mask_list[i], size=(w0, w0), mode='bilinear',
                                         align_corners=True)
                tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
                s = self.theta(tmp_supp_feat_4)
                s = F.normalize(s, 2, 1)
                if self.args.down_size > 0:
                    s = F.interpolate(s, size=(s.shape[-2] // self.args.down_size, s.shape[-1] // self.args.down_size),
                                      mode='bilinear', align_corners=True)
                bs, cs, ws, _ = s.size()
                x_patch = F.unfold(s, kernel_size=multi_scales[j], stride=1,
                                   padding=paddings[j])
                x_patch = x_patch.permute(0, 2, 1)
                x_patch = x_patch.view(bs, ws * ws, cs, multi_scales[j] * multi_scales[j])
                x_patch = x_patch.view(bs, ws * ws, cs, multi_scales[j], multi_scales[j])
                output_list = []
                for m in range(bs):
                    response = F.conv2d(q[m].unsqueeze(0), x_patch[m], stride=1,
                                        padding=paddings[j])
                    output_list.append(response)
                output = torch.cat(output_list, 0)
                output = output.contiguous().view(bs, -1, q.shape[2] * q.shape[3])
                if self.args.multi_nsm:
                    output = self.nsm[j](output.permute(0, 2, 1).unsqueeze(1))
                else:
                    output = self.nsm(output.permute(0, 2, 1).unsqueeze(1))
                output = output.view(bs, 1, q.shape[2], q.shape[3])
                multi_scales_mask_list[j].append(output)

            multi_scales_mask = torch.cat(multi_scales_mask_list[j], 1).mean(1).unsqueeze(1)
            response_mask_list.append(multi_scales_mask)
        response_mask = torch.cat(response_mask_list, 1)

        if self.args.use_mid_nsm:
            q = full_query_feat.clone()
            q = self.theta_mid(q)
            q = F.normalize(q, 2, 1)
            multi_scales = self.args.mid_multi_scales
            paddings = self.args.mid_paddings
            multi_scales_mask_list = [[], [], []]
            mid_response_mask_list = []
            if self.args.query_down_size > 1:
                q = F.interpolate(q, size=(
                q.shape[-2] // self.args.query_down_size, q.shape[-1] // self.args.query_down_size), mode='bilinear',
                                  align_corners=True)
            for j in range(len(multi_scales)):
                cosine_eps = 1e-7
                for i, tmp_supp_feat in enumerate(full_supp_list):
                    b0, c0, w0, _ = tmp_supp_feat.size()
                    tmp_mask = F.interpolate(mask_list[i], size=(w0, w0), mode='bilinear',
                                             align_corners=True)
                    s = tmp_supp_feat * tmp_mask
                    s = self.theta_mid(s)
                    s = F.normalize(s, 2, 1)
                    if self.args.down_size > 0:
                        s = F.interpolate(s,
                                          size=(s.shape[-2] // self.args.down_size, s.shape[-1] // self.args.down_size),
                                          mode='bilinear', align_corners=True)
                    bs, cs, ws, _ = s.size()
                    x_patch = F.unfold(s, kernel_size=multi_scales[j], stride=1,
                                       padding=paddings[j])
                    x_patch = x_patch.permute(0, 2, 1)
                    x_patch = x_patch.view(bs, ws * ws, cs, multi_scales[j] * multi_scales[j])
                    x_patch = x_patch.view(bs, ws * ws, cs, multi_scales[j], multi_scales[j])
                    output_list = []
                    for m in range(bs):
                        response = F.conv2d(q[m].unsqueeze(0), x_patch[m], stride=1,
                                            padding=paddings[j])
                        output_list.append(response)
                    output = torch.cat(output_list, 0)
                    output = output.contiguous().view(bs, -1, q.shape[2] * q.shape[3])

                    if self.args.multi_nsm:
                        output = self.nsm[j](output.permute(0, 2, 1).unsqueeze(1))
                    else:
                        output = self.nsm(output.permute(0, 2, 1).unsqueeze(1))
                    output = output.view(bs, 1, q.shape[2], q.shape[3])
                    multi_scales_mask_list[j].append(output)
                multi_scales_mask = torch.cat(multi_scales_mask_list[j], 1).mean(1).unsqueeze(1)
                mid_response_mask_list.append(multi_scales_mask)
            mid_response_mask = torch.cat(mid_response_mask_list, 1)

        out_list = []
        pyramid_feat_list = []


        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
            response_mask_bin = F.interpolate(response_mask, size=(bin, bin), mode='bilinear',
                                              align_corners=True)
            if self.args.use_mid_nsm:
                mid_response_mask_bin = F.interpolate(mid_response_mask, size=(bin, bin), mode='bilinear',
                                                      align_corners=True)
                response_mask_bin = torch.cat([response_mask_bin, mid_response_mask_bin], 1)

            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
            if not self.args.real_multi_supp:
                merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, response_mask_bin],
                                           1)
                merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear',
                                             align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](
                merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())
            aux_loss = torch.zeros_like(main_loss).cuda()

            for idx_k in range(len(out_list)):  # 4
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear',
                                          align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())
            aux_loss = aux_loss / len(out_list)
            return out.max(1)[1], main_loss, aux_loss
        else:
            return out
