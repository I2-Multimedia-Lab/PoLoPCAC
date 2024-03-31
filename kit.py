import os
import math
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tqdm import tqdm
from plyfile import PlyElement, PlyData
from pyntcloud import PyntCloud
from pytorch3d.ops.knn import knn_gather, knn_points


#core transformation function
def transformRGBToYCoCg(bitdepth, rgb):
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    co = r - b
    t = b + (co >> 1) # co >>1 i.e. co // 2
    cg = g - t
    y = t + (cg >> 1)

    offset = 1 << bitdepth # 2^bitdepth

    # NB: YCgCoR needs extra 1-bit for chroma
    return np.column_stack((y,  co + offset, cg + offset))


def transformYCoCgToRGB(bitDepth,  ycocg):

    offset = 1 << bitDepth
    y0 = ycocg[:,0]
    co = ycocg[:,1] - offset
    cg = ycocg[:,2] - offset

    t = y0 - (cg >> 1)

    g = cg + t
    b = t - (co >> 1)
    r = co + b

    maxVal = (1 << bitDepth) - 1
    r = np.clip(r, 0, maxVal)
    g = np.clip(g, 0, maxVal)
    b = np.clip(b, 0, maxVal)

    return np.column_stack((r,g,b))


def read_point_cloud_ycocg(filepath):
    pc = PyntCloud.from_file(filepath)
    try:
        cols=['x', 'y', 'z','red', 'green', 'blue']
        points=pc.points[cols].values
    except:
        cols = ['x', 'y', 'z', 'r', 'g', 'b']
        points = pc.points[cols].values
    color = points[:, 3:].astype(np.int16)
    color = transformRGBToYCoCg(8, color)
    # color: int
    # y channel: 0~255
    # co channel: 0~511 (1~511 in our dataset)
    # cg channel: 0~511 (34~476 in our dataset)
    points[:, 3:] = color.astype(float)
    return points


def save_point_cloud_ycocg(pc, path):
    color = pc[:, 3:]
    color = np.round(color).astype(np.int16) # 务必 round 后 再加 astype
    color = transformYCoCgToRGB(8, color)

    pc = pd.DataFrame(pc, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
    pc[['red','green','blue']] = np.round(color).astype(np.uint8)
    cloud = PyntCloud(pc)
    cloud.to_file(path)


def read_point_cloud_reflactance(filepath):
    plydata = PlyData.read(filepath)
    pc = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'], plydata['vertex']['reflectance'])))).astype(np.float32)        
    pc[:, 3:] = pc[:, 3:] / 100
    return pc


def save_point_cloud_reflactance(pc, path, to_rgb=False):

    if to_rgb:
        cmap = plt.get_cmap('jet')
        color = np.round(cmap(pc[:, 3])[:, :3] * 255)
        pc = np.hstack((pc[:, :3], color))
        pc = pd.DataFrame(pc, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
        pc[['red','green','blue']] = np.round(np.clip(pc[['red','green','blue']], 0, 255)).astype(np.uint8)
        cloud = PyntCloud(pc)
        cloud.to_file(path)
    else:
        scan = pc
        vertex = np.array(
            [(scan[i,0], scan[i,1], scan[i,2], scan[i,3]*100) for i in range(scan.shape[0])],
            dtype=[
                ("x", np.dtype("float32")), 
                ("y", np.dtype("float32")), 
                ("z", np.dtype("float32")),
                ("reflectance", np.dtype("uint8")),
            ]
        )
        PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        output_pc = PlyElement.describe(vertex, "vertex")
        output_pc = PlyData([output_pc])
        output_pc.write(path)


def read_point_clouds_ycocg(file_path_list, bar=True):
    print('loading point clouds...')
    with multiprocessing.Pool() as p:
        if bar:
            pcs = list(tqdm(p.imap(read_point_cloud_ycocg, file_path_list, 32), total=len(file_path_list)))
        else:
            pcs = list(p.imap(read_point_cloud_ycocg, file_path_list, 32))
    return pcs



def n_scale_ball(grouped_xyz):
    B, N, K, _ = grouped_xyz.shape

    longest = (grouped_xyz**2).sum(dim=-1).sqrt().max(dim=-1)[0]
    scaling = (1) / longest
    
    grouped_xyz = grouped_xyz * scaling.view(B, N, 1, 1)

    return grouped_xyz


class MLP(nn.Module):
    def __init__(self, in_channel, mlp, relu, bn):
        super(MLP, self).__init__()

        mlp.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlp) - 1):
            if relu[i]:
                if bn[i]:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlp[i], mlp[i+1], 1),
                        nn.BatchNorm2d(mlp[i+1]),
                        nn.ReLU(),
                    )
                else:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlp[i], mlp[i+1], 1),
                        nn.ReLU(),
                    )
            else:
                mlp_Module = nn.Sequential(
                    nn.Conv2d(mlp[i], mlp[i+1], 1),
                )
            self.mlp_Modules.append(mlp_Module)


    def forward(self, points, squeeze=False):
        """
        Input:
            points: input points position data, [B, C, N]
        Return:
            points: feature data, [B, D, N]
        """
        if squeeze:
            points = points.unsqueeze(-1) # [B, C, N, 1]
        
        for m in self.mlp_Modules:
            points = m(points)
        # [B, D, N, 1]
        
        if squeeze:
            points = points.squeeze(-1) # [B, D, N] 

        return points
    

class QueryMaskedAttention(nn.Module):
    def __init__(self, channel):
        super(QueryMaskedAttention, self).__init__()
        self.channel = channel
        self.k_mlp = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.v_mlp = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.pe_multiplier, self.pe_bias = True, True
        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
            )
        self.weight_encoding = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
        )
        self.residual_emb = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, grouped_xyz, grouped_feature):

        key = self.k_mlp(grouped_feature) # B, C, K, M
        value = self.v_mlp(grouped_feature) # B, C, K, M

        relation_qk = key #  - query
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(grouped_xyz)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(grouped_xyz)
            relation_qk = relation_qk + peb
            value = value + peb

        weight  = self.weight_encoding(relation_qk)
        score = self.softmax(weight) # B, C, K, M

        feature = score*value # B, C, K, M
        feature = self.residual_emb(feature) # B, C, K, M

        return feature


class PT(nn.Module):
    def __init__(self, in_channel, out_channel, n_layers):
        super(PT, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_layers = n_layers
        self.sa_ls, self.sa_emb_ls = nn.ModuleList(), nn.ModuleList()
        self.linear_in = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        for i in range(n_layers):
            self.sa_emb_ls.append(nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1),
                nn.ReLU(),
            ))
            self.sa_ls.append(QueryMaskedAttention(out_channel))
    def forward(self, groped_geo, grouped_attr):
        """
        Input:
            groped_geo: input points position data, [B, M, K, 3]
            groped_attr: input points feature data, [B, M, K, 3]
        Return:
            feature: output feature data, [B, M, C]
        """
        groped_geo, grouped_attr = groped_geo.permute((0, 3, 2, 1)), grouped_attr.permute((0, 3, 2, 1)) # B, _, K, M
        feature = self.linear_in(grouped_attr)
        for i in range(self.n_layers):
            identity = feature
            feature = self.sa_emb_ls[i](feature)
            output = self.sa_ls[i](groped_geo, feature)
            feature = output + identity
        feature = feature.sum(dim=2).transpose(1, 2)
        return feature


def get_cdf(mu, sigma):
    M, d = sigma.shape
    mu = mu.unsqueeze(-1).repeat(1, 1, 256)
    sigma = sigma.unsqueeze(-1).repeat(1, 1, 256).clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    flag = torch.arange(0, 256).to(sigma.device).view(1, 1, 256).repeat((M, d, 1))
    cdf = gaussian.cdf(flag + 0.5)

    spatial_dimensions = cdf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=cdf.dtype, device=cdf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    return cdf_with_0


def get_cdf_ycocg(mu, sigma):
    M, d = sigma.shape
    mu = mu.unsqueeze(-1).repeat(1, 1, 512)
    sigma = sigma.unsqueeze(-1).repeat(1, 1, 512).clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    flag = torch.arange(0, 512).to(sigma.device).view(1, 1, 512).repeat((M, d, 1))
    cdf = gaussian.cdf(flag + 0.5)

    spatial_dimensions = cdf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=cdf.dtype, device=cdf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    return cdf_with_0


def feature_probs_based_mu_sigma(feature, mu, sigma):
    sigma = sigma.clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
    return total_bits, probs


def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8


def _convert_to_int_and_normalize(cdf_float, needs_normalization):
  """Convert floatingpoint CDF to integers. See README for more info.

  The idea is the following:
  When we get the cdf here, it is (assumed to be) between 0 and 1, i.e,
    cdf \in [0, 1)
  (note that 1 should not be included.)
  We now want to convert this to int16 but make sure we do not get
  the same value twice, as this would break the arithmetic coder
  (you need a strictly monotonically increasing function).
  So, if needs_normalization==True, we multiply the input CDF
  with 2**16 - (Lp - 1). This means that now,
    cdf \in [0, 2**16 - (Lp - 1)].
  Then, in a final step, we add an arange(Lp), which is just a line with
  slope one. This ensure that for sure, we will get unique, strictly
  monotonically increasing CDFs, which are \in [0, 2**16)
  """
  Lp = cdf_float.shape[-1]
  factor = torch.tensor(
    2, dtype=torch.float32, device=cdf_float.device).pow_(16)
  new_max_value = factor
  if needs_normalization:
    new_max_value = new_max_value - (Lp - 1)
  cdf_float = cdf_float.mul(new_max_value)
  cdf_float = cdf_float.round()
  cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
  if needs_normalization:
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
  return cdf
