"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    # @info 计算深度分布和图像特征的外积，获得视椎特征
    # @param x 骨干网络提取的特征图
    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x) #通过depthnet网络提取分类深度分布特征

        # 取特征图的前D维，即分类深度分布特征，进行softmax归一化，转成深度概率图（深度分布）
        depth = self.get_depth_dist(x[:, :self.D])
        # 计算深度分布和表示颜色的特征图的后C维的外积，获得视椎特征
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        # 给定BEV网格在三个维度上的边界和步长如下
        #   xbound=[-50.0, 50.0, 0.5]
        #   ybound=[-50.0, 50.0, 0.5]
        #   zbound=[-10.0, 10.0, 20.0]
        # dx是每个BEV网格在x/y/z三个维度上的边长，即步长[0.50, 0.50, 20.00]
        # bx是在x/y/z三个维度上第一个网格中心点坐标[-49.75, -49.75, 0.00]
        # nx是在x/y/z三个维度上的网格数：[200, 200, 1]
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    # @info 创建视椎体点云坐标（像素坐标）
    # @return 返回形如[D,H,W,3]的视椎体点云坐标，其中D为深度，H为图像高，W为图像宽，3为坐标轴
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    # @info 将视椎体点云从像素坐标系坐标转到雷达系
    # @param rots 相机到雷达的旋转矩阵
    # @param trans 相机到雷达的平移向量
    # @param intrins 相机内参
    # @param post_rots 图像增强变换矩阵-旋转
    # @param post_trans 图像增强变换矩阵-平移
    # @return 返回转到雷达系的视椎体点云坐标
    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    # @info 进行Lift操作，获得视椎特征
    # @param x 骨干网络提取的特征图
    # @return 视椎特征（点云）
    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        # 将特征图的前两维合并，变成[B*N,C,H,W]
        x = x.view(B*N, C, imH, imW)
        # 进行Lift操作，获得视椎特征
        x = self.camencode(x)
        # 将视椎特征的维度还原成[B,N,C,D,H,W]
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        # 将视椎特征图的维度进行转置，变成[batch_size、图像数、深度、img高、img宽、通道数]，即[B,N,D,H,W,C]
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    # @info 进行Splat操作，通过sum-pooling的方式，获得BEV特征
    # @param geom_feats 视椎点云坐标，或几何特征
    # @param x Lift操作获得的视椎特征
    # @return BEV特征
    def voxel_pooling(self, geom_feats, x):

        # 取得视椎特征的形状和特征点的总数
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # 把视椎特征完全展平
        # flatten x
        x = x.reshape(Nprime, C)

        # 把视椎点云坐标转成BEV网格坐标，并展平
        # 需要注意的是，BEV网格的形状是[0.5, 0.5, 20]，BEV网格的数量是[200,200,1]，也就是说Z轴方向上只有一个网格，高度20米
        # 视椎点云坐标转成BEV网格坐标后，Z轴方向上的坐标值都是0，也就是说所有的几何特征都在Z=0的平面上
        # BEV网格实际是200×200个柱子，即BEV Pillar，每个柱子高度20米，柱子的宽度和长度都是0.5米
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)

        # 创建批次索引，拼接到展平的视椎点云坐标后面，拼接后的形状为(Nprime, 4)，前三列分别是x,y,z，第4列是批次索引
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # 过滤掉超出边界的视椎点云坐标和视椎特征
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # 按照BEV视角下B、D、W、H的顺序，对视椎点云坐标和视椎特征进行升序排序
        # D表示Z轴，W表示Y轴，H表示X轴
        # 排序完成后，属于同一个BEV Pillar的视椎特征的ranks值是相同的
        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 通过geom_feats中的网格索引，将feats中的特征值累加到对应的BEV网格中
        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # 将累加后的特征值，按照B、C、Z、X、Y的顺序，转成BEV特征图
        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # 折叠Z坐标
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # 获得视椎体点云坐标
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        #进行Lift操作，获得视椎特征
        x = self.get_cam_feats(x)
        # 进行Splat操作，完成柱体池化，获得BEV特征
        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
