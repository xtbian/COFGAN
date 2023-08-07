from collections import OrderedDict
import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_3D import SEGating

from .base_nets import BaseSequenceGenerator, BaseSequenceDiscriminator
from codes.utils.net_utils import space_to_depth, backward_warp, get_upsampling_func
from codes.utils.net_utils import initialize_weights
from codes.utils.data_utils import float32_to_uint8
from codes.metrics.model_summary import register, parse_model_info


def joinTensors(X1, X2, type="concat"):
    if type == "concat":
        return torch.cat([X1, X2], dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


# ====================== generator modules ====================== #

# U-Net运动估计网络
class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):
        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv3D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose", batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode == "transpose":
            self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                SEGating(out_ch)
                ]
            )

        else:
            self.upconv = nn.ModuleList(
                [nn.Upsample(mode='trilinear', scale_factor=(1, 2, 2), align_corners=False),
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1),
                SEGating(out_ch)
                ]
                )

        if batchnorm:
            self.upconv += [nn.BatchNorm3d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    SEGating(out_ch)
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv2D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose", batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode == "transpose":
            self.upconv = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]

        else:
            self.upconv = [
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
            ]

        if batchnorm:
            self.upconv += [nn.BatchNorm2d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)

class MEnet(nn.Module):
    def __init__(self, block, n_inputs, n_outputs, batchnorm=False, joinType="concat", upmode="transpose"):
        super().__init__()

        nf = [512, 256, 128, 64]
        out_channels = 3 * n_outputs
        self.joinType = joinType
        self.n_outputs = n_outputs

        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, True)

        unet_3D = importlib.import_module('models.networks.resnet_3D')
        if n_outputs > 1:
            unet_3D.useBias = True
        self.encoder = getattr(unet_3D , block)(pretrained=False, bn=batchnorm)

        self.decoder = nn.Sequential(
            Conv_3d(nf[0], nf[1], kernel_size=3, padding=1, bias=True, batchnorm=batchnorm),
            upConv3D(nf[1] * growth, nf[2], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),
            upConv3D(nf[2] * growth, nf[3], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),
            # Conv_3d(nf[3] * growth, nf[3], kernel_size=3, padding=1, bias=True, batchnorm=batchnorm),
            upConv3D(nf[3] * growth, nf[3], kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), upmode=upmode, batchnorm=batchnorm),
            # Conv_3d(nf[3] * growth, nf[3], kernel_size=3, padding=1, bias=True, batchnorm=batchnorm)
            Conv_3d(nf[3], nf[3], kernel_size=3, padding=1, bias=True, batchnorm=batchnorm)
        )

        self.feature_fuse = Conv_2d(nf[2], nf[3], kernel_size=1, stride=1, batchnorm=batchnorm)
        #融合时间模块, 3D卷积变为2D   nf[3] * n_inputs为原来输入的通道数，但与3D conv输出通道数不一致，因此将 * n_inputs 去掉，通道数均变成64

        # self.outconv = nn.Sequential(
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(nf[3], out_channels, kernel_size=7, stride=1, padding=0)
        # )   #填充卷积用来预测
        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, bias=True)
        )


    def forward(self, lr_prev, lr_curr):

        lr_curr = torch.unsqueeze(lr_curr, dim=2)                                       #扩展维度
        lr_prev = torch.unsqueeze(lr_prev, dim=2)
        mean_ = lr_curr.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)#批量归一化Batch mean normalization
        mean__ = lr_prev.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        lr_curr = lr_curr - mean_
        lr_prev = lr_prev - mean__

        image = torch.cat([lr_prev, lr_curr], dim=2)

        x_0, x_1, x_2, x_3, x_4 = self.encoder(image)   #拼接维度

        dx_3 = self.lrelu(self.decoder[0](x_4))
        dx_3 = joinTensors(dx_3, x_3, type=self.joinType)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = joinTensors(dx_2, x_2, type=self.joinType)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = joinTensors(dx_1, x_1, type=self.joinType)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        # dx_0 = joinTensors(dx_0, x_0, type=self.joinType)

        dx_out = self.lrelu(self.decoder[4](dx_0))
        dx_out = torch.cat(torch.unbind(dx_out, 2), 1)  #unbind函数将指定维度移除，对dx_out第3维切片，不改变tensor的shape，只是返回移除的维度 cat()函数按照指定维度将向量拼接
        #Tensor(18,128,32,32)

        out = self.lrelu(self.feature_fuse(dx_out)) #(18,64,32,32)
        # out = self.outconv(out)

        # out = torch.split(out, dim=1, split_size_or_sections=3) #切分tensor
        # mean_ = mean_.squeeze(2)
        # mean__ = mean__.squeeze(2)
        # out = [o + mean_ for o in out]
        # F_1_out = F.interpolate(torch.cat(torch.unbind(dx_1, 2),2), scale_factor=2, mode='bilinear', align_corners=False)#生成光流图
        # F_2_out = F.interpolate(F_1_out, scale_factor=2, mode='bilinear', align_corners=False)
        # F_3_out = F.interpolate(F_2_out, scale_factor=2, mode='bilinear', align_corners=False)
        #
        # out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity
        return out


class ResidualBlock(nn.Module):
    """ Residual block without batch normalization
    """

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SRNet(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc, out_nc, nf, nb, upsample_func, scale):
        super(SRNet, self).__init__()

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling blocks
        conv_up = [
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True)]

        if scale == 4:
            conv_up += [
                nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
                nn.ReLU(inplace=True)]

        self.conv_up = nn.Sequential(*conv_up)

        # output conv.
        self.conv_out = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # upsampling function
        self.upsample_func = upsample_func

    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(s*s*c)hw
        """

        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        out = self.resblocks(out)
        out = self.conv_up(out)
        out = self.conv_out(out)
        out += self.upsample_func(lr_curr)

        return out


class FRNet(BaseSequenceGenerator):
    """ Frame-recurrent network: https://arxiv.org/abs/1801.04590
    """

    def __init__(self, in_nc, out_nc, nf, nb, degradation, scale):
        super(FRNet, self).__init__()

        self.scale = scale

        # get upsampling function according to degradation type
        self.upsample_func = get_upsampling_func(self.scale, degradation)

        model_choices = ["unet_18", "unet_34"]
        # define MEnet & srnet
        self.menet = MEnet(model_choices[0], in_nc, out_nc)
        self.srnet = SRNet(in_nc, out_nc, nf, nb, self.upsample_func, self.scale)

    def forward(self, lr_data, device=None):
        if self.training:
            out = self.forward_sequence(lr_data)
        else:
            out = self.infer_sequence(lr_data, device)

        return out

    def forward_sequence(self, lr_data):
        """
            Parameters:
                :param lr_data: lr data in shape ntchw
        """

        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # calculate optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_flow = self.menet(lr_curr, lr_prev)  # n*(t-1),2,h,w

        # upsample lr flows
        hr_flow = self.scale * self.upsample_func(lr_flow)
        hr_flow = hr_flow.view(n, (t - 1), 64, hr_h, hr_w)  #调整Tensor的shape 和reshape函数类似

        # compute the first hr data
        hr_data = []
        hr_prev = self.srnet(
            lr_data[:, 0, ...],
            torch.zeros(n, (self.scale**2)*c, lr_h, lr_w, dtype=torch.float32,
                        device=lr_data.device))
        hr_data.append(hr_prev)

        # compute the remaining hr data
        for i in range(1, t):
            # warp hr_prev
            hr_prev_warp = backward_warp(hr_prev, hr_flow[:, i - 1, ...])

            # compute hr_curr
            hr_curr = self.srnet(
                lr_data[:, i, ...],
                space_to_depth(hr_prev_warp, self.scale))

            # save and update
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w

        # construct output dict
        ret_dict = {
            'hr_data': hr_data,  # n,t,c,hr_h,hr_w
            'hr_flow': hr_flow,  # n,t,2,hr_h,hr_w
            'lr_prev': lr_prev,  # n(t-1),c,lr_h,lr_w
            'lr_curr': lr_curr,  # n(t-1),c,lr_h,lr_w
            'lr_flow': lr_flow,  # n(t-1),2,lr_h,lr_w
        }

        return ret_dict

    def step(self, lr_curr, lr_prev, hr_prev):
        """
            Parameters:
                :param lr_curr: the current lr data in shape nchw
                :param lr_prev: the previous lr data in shape nchw
                :param hr_prev: the previous hr data in shape nc(sh)(sw)
        """

        # estimate lr flow (lr_curr -> lr_prev)
        lr_flow = self.menet(lr_curr, lr_prev)

        # pad if size is not a multiple of 8
        pad_h = lr_curr.size(2) - lr_curr.size(2)//8*8
        pad_w = lr_curr.size(3) - lr_curr.size(3)//8*8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), 'reflect')

        # upsample lr flow
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)

        # warp hr_prev
        hr_prev_warp = backward_warp(hr_prev, hr_flow)

        # compute hr_curr
        hr_curr = self.srnet(lr_curr, space_to_depth(hr_prev_warp, self.scale))

        return hr_curr

    def infer_sequence(self, lr_data, device):
        """
            Parameters:
                :param lr_data: torch.FloatTensor in shape tchw
                :param device: torch.device

                :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # set params
        tot_frm, c, h, w = lr_data.size()
        s = self.scale

        # forward
        hr_seq = []
        lr_prev = torch.zeros(1, c, h, w, dtype=torch.float32).to(device)
        hr_prev = torch.zeros(1, c, s*h, s*w, dtype=torch.float32).to(device)

        with torch.no_grad():
            for i in range(tot_frm):
                lr_curr = lr_data[i: i + 1, ...].to(device)
                hr_curr = self.step(lr_curr, lr_prev, hr_prev)
                lr_prev, hr_prev = lr_curr, hr_curr

                hr_frm = hr_curr.squeeze(0).cpu().numpy()  # chw|rgb|uint8
                hr_seq.append(float32_to_uint8(hr_frm))

        return np.stack(hr_seq).transpose(0, 2, 3, 1)  # thwc

    def generate_dummy_data(self, lr_size, device):
        c, lr_h, lr_w = lr_size
        s = self.scale

        # generate dummy input data
        lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32).to(device)
        lr_prev = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32).to(device)
        hr_prev = torch.rand(1, c, s*lr_h, s*lr_w, dtype=torch.float32).to(device)

        data_list = [lr_curr, lr_prev, hr_prev]
        return data_list

    def profile(self, lr_size, device):
        gflops_dict, params_dict = OrderedDict(), OrderedDict()

        # generate dummy input data
        lr_curr, lr_prev, hr_prev = self.generate_dummy_data(lr_size, device)

        # profile module 1: flow estimation module
        lr_flow = register(self.menet, [lr_curr, lr_prev])
        gflops_dict['MENet'], params_dict['MENet'] = parse_model_info(self.menet)

        # profile module 2: sr module
        pad_h = lr_curr.size(2) - lr_curr.size(2)//8*8
        pad_w = lr_curr.size(3) - lr_curr.size(3)//8*8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), 'reflect')
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)
        hr_prev_warp = backward_warp(hr_prev, hr_flow)
        _ = register(self.srnet, [lr_curr, space_to_depth(hr_prev_warp, self.scale)])
        gflops_dict['SRNet'], params_dict['SRNet'] = parse_model_info(self.srnet)

        return gflops_dict, params_dict


# ====================== discriminator modules ====================== #
class DiscriminatorBlocks(nn.Module):
    def __init__(self):
        super(DiscriminatorBlocks, self).__init__()

        self.block1 = nn.Sequential(  # /2
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.block2 = nn.Sequential(  # /4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.block3 = nn.Sequential(  # /8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.block4 = nn.Sequential(  # /16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        feature_list = [out1, out2, out3, out4]

        return out4, feature_list


class SpatioTemporalDiscriminator(BaseSequenceDiscriminator):
    """ Spatio-Temporal discriminator proposed in TecoGAN
    """

    def __init__(self, in_nc, spatial_size, tempo_range, degradation, scale):
        super(SpatioTemporalDiscriminator, self).__init__()

        # basic settings
        mult = 3  # (conditional triplet, input triplet, warped triplet)
        self.spatial_size = spatial_size
        self.tempo_range = tempo_range
        assert self.tempo_range == 3, 'currently only support 3 as tempo_range'
        self.scale = scale

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc*tempo_range*mult, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        # discriminator block
        self.discriminator_block = DiscriminatorBlocks()  # downsample 16x

        # classifier
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

        # get upsampling function according to degradation type
        self.upsample_func = get_upsampling_func(self.scale, degradation)

    def forward(self, data, args_dict):
        out = self.forward_sequence(data, args_dict)
        return out

    def forward_sequence(self, data, args_dict):
        """
            :param data: should be either hr_data or gt_data
            :param args_dict: a dict including data/config required here
        """

        # === set params === #
        net_G = args_dict['net_G']
        lr_data = args_dict['lr_data']
        bi_data = args_dict['bi_data']
        hr_flow = args_dict['hr_flow']

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, hr_h, hr_w = data.size()

        s_size = self.spatial_size
        t = t // 3 * 3  # discard other frames
        n_clip = n * t // 3  # total number of 3-frame clips in all batches

        c_size = int(s_size * args_dict['crop_border_ratio'])
        n_pad = (s_size - c_size) // 2

        # === compute forward & backward flow === #
        if 'hr_flow_merge' not in args_dict:
            if args_dict['use_pp_crit']:
                hr_flow_bw = hr_flow[:, 0:t:3, ...]  # e.g., frame1 -> frame0
                hr_flow_idle = torch.zeros_like(hr_flow_bw)
                hr_flow_fw = hr_flow.flip(1)[:, 1:t:3, ...]
            else:
                lr_curr = lr_data[:, 1:t:3, ...]
                lr_curr = lr_curr.reshape(n_clip, c, lr_h, lr_w)

                lr_next = lr_data[:, 2:t:3, ...]
                lr_next = lr_next.reshape(n_clip, c, lr_h, lr_w)

                # compute forward flow
                lr_flow_fw = net_G.fnet(lr_curr, lr_next)
                hr_flow_fw = self.scale * self.upsample_func(lr_flow_fw)

                hr_flow_bw = hr_flow[:, 0:t:3, ...]  # e.g., frame1 -> frame0
                hr_flow_idle = torch.zeros_like(hr_flow_bw)  # frame1 -> frame1
                hr_flow_fw = hr_flow_fw.view(n, t // 3, 2, hr_h, hr_w)  # frame1 -> frame2

            # merge bw/idle/fw flows
            hr_flow_merge = torch.stack(
                [hr_flow_bw, hr_flow_idle, hr_flow_fw], dim=2)  # n,t//3,3,2,h,w

            # reshape and stop gradient propagation
            hr_flow_merge = hr_flow_merge.view(n_clip * 3, 64, hr_h, hr_w).detach()

        else:
            # reused data to reduce computations
            hr_flow_merge = args_dict['hr_flow_merge']

        # === build up inputs for D (3 parts) === #
        # part 1: bicubic upsampled data (conditional inputs)
        cond_data = bi_data[:, :t, ...].reshape(n_clip, 3, c, hr_h, hr_w)
        # note: permutation is not necessarily needed here, it's just to keep
        #       the same impl. as TecoGAN-Tensorflow (i.e., rrrgggbbb)
        cond_data = cond_data.permute(0, 2, 1, 3, 4)
        cond_data = cond_data.reshape(n_clip, c * 3, hr_h, hr_w)

        # part 2: original data
        orig_data = data[:, :t, ...].reshape(n_clip, 3, c, hr_h, hr_w)
        orig_data = orig_data.permute(0, 2, 1, 3, 4)
        orig_data = orig_data.reshape(n_clip, c * 3, hr_h, hr_w)

        # part 3: warped data
        warp_data = backward_warp(
            data[:, :t, ...].reshape(n * t, c, hr_h, hr_w), hr_flow_merge)
        warp_data = warp_data.view(n_clip, 3, c, hr_h, hr_w)
        warp_data = warp_data.permute(0, 2, 1, 3, 4)
        warp_data = warp_data.reshape(n_clip, c * 3, hr_h, hr_w)
        # remove border to increase training stability as proposed in TecoGAN
        warp_data = F.pad(
            warp_data[..., n_pad: n_pad + c_size, n_pad: n_pad + c_size],
            (n_pad,) * 4, mode='constant')

        # combine 3 parts together
        input_data = torch.cat([orig_data, warp_data, cond_data], dim=1)

        # === classify === #
        out = self.conv_in(input_data)
        out, feature_list = self.discriminator_block(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        pred = out, feature_list

        # construct output dict (return other data beside pred)
        ret_dict = {
            'hr_flow_merge': hr_flow_merge
        }

        return pred, ret_dict


class SpatialDiscriminator(BaseSequenceDiscriminator):
    """ Spatial discriminator
    """

    def __init__(self, in_nc, spatial_size, use_cond):
        super(SpatialDiscriminator, self).__init__()

        # basic settings
        self.use_cond = use_cond  # whether to use conditional input
        mult = 2 if self.use_cond else 1
        tempo_range = 1

        # input conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc*tempo_range*mult, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        # discriminator block
        self.discriminator_block = DiscriminatorBlocks()  # /16

        # classifier
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

    def forward(self, data, args_dict):
        out = self.forward_sequence(data, args_dict)
        return out

    def step(self, x):
        out = self.conv_in(x)
        out, feature_list = self.discriminator_block(out)

        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out, feature_list

    def forward_sequence(self, data, args_dict):
        # === set params === #
        n, t, c, hr_h, hr_w = data.size()
        data = data.view(n * t, c, hr_h, hr_w)

        # === build up inputs for net_D === #
        if self.use_cond:
            bi_data = args_dict['bi_data'].view(n * t, c, hr_h, hr_w)
            input_data = torch.cat([bi_data, data], dim=1)
        else:
            input_data = data

        # === classify === #
        pred = self.step(input_data)

        # construct output dict (nothing to return)
        ret_dict = {}

        return pred, ret_dict
