import os
import time
import argparse

import numpy as np

from glob import glob
from tqdm import tqdm

import torch
import torchac
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

import kit
from net import Network

torch.cuda.manual_seed(1)
torch.manual_seed(1)
np.random.seed(1)


parser = argparse.ArgumentParser(
    prog='reflectance_decompress.py',
    description='Decompress Point Cloud Reflectance Attributes.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ckpt', required=True, help='Trained ckeckpoint file.')
parser.add_argument('--compressed_path', required=True, help='Compressed file saving directory.')
parser.add_argument('--decompressed_path', required=True, help='Decompressed file saving directory.')

parser.add_argument('--local_region', type=int, help='', default=8)
parser.add_argument('--granularity', type=int, help='', default=2**14)
parser.add_argument('--init_ratio', type=int, help='', default=128)
parser.add_argument('--expand_ratio', type=int, help='', default=2)
parser.add_argument('--prg_seed', type=int, help='', default=2147483647)

args = parser.parse_args()


if not os.path.exists(args.decompressed_path):
    os.makedirs(args.decompressed_path)

comp_glob = os.path.join(args.compressed_path, '*.c.bin')
files = np.array(glob(comp_glob, recursive=True))
np.random.shuffle(files)
files = files[:]

net = Network(local_region=args.local_region, granularity=args.granularity, init_ratio=args.init_ratio, expand_ratio=args.expand_ratio)
net.load_state_dict(torch.load(args.ckpt))
net = torch.compile(net, mode='max-autotune')
net.cuda().eval()

# warm up our model
# since the very first step of network is extremely slow...
_ = net.mu_sigma_pred(net.pt(torch.rand((1, 32, 8, 3)).cuda(), torch.rand((1, 32, 8, 3)).cuda()))

dec_times = []
with torch.no_grad():
    for comp_c_f in tqdm(files):
        fname = os.path.split(comp_c_f)[-1].split('.c.bin')[0]
        geo_f_path = os.path.join(args.compressed_path, fname+'.geo.bin')

        batch_x_geo = torch.tensor(np.fromfile(geo_f_path, dtype=np.float32)).view(1, -1, 3)
        context_attr_base = torch.tensor(np.fromfile(comp_c_f, dtype=np.uint8)).view(1, -1, 1)

        torch.cuda.synchronize()
        TIME_STAMP = time.time()

        _, N, _ = batch_x_geo.shape

        base_size = min(N//args.init_ratio, args.granularity)
        window_size = base_size
        cursor = base_size
        i=0
        while cursor < N:
            window_size = min(window_size*args.expand_ratio, args.granularity)
            
            context_geo = batch_x_geo[:, :cursor, :].cuda()
            target_geo = batch_x_geo[:, cursor:cursor+window_size, :].cuda()
            cursor += window_size
            
            context_attr = context_attr_base.float().cuda() / 100
            context_attr = context_attr.repeat((1, 1, 3))

            _, idx, context_grouped_geo = knn_points(target_geo, context_geo, K=net.local_region, return_nn=True)
            context_grouped_attr = knn_gather(context_attr, idx)

            context_grouped_geo = context_grouped_geo - target_geo.view(1, -1, 1, 3)
            context_grouped_geo = kit.n_scale_ball(context_grouped_geo)

            feature = net.pt(context_grouped_geo, context_grouped_attr)
            mu_sigma = net.mu_sigma_pred(feature)
            mu, sigma = mu_sigma[:, :, :3]+0.5, torch.exp(mu_sigma[:, :, 3:])

            cdf = kit.get_cdf_reflactance(mu[0]*100, sigma[0]*32)
            cdf = cdf[:, 0, :]
            comp_f = os.path.join(args.compressed_path, fname+f'.{i}.bin')
            with open(comp_f, 'rb') as fin:
                byte_stream = fin.read()
            # decomp_attr = torchac.decode_float_cdf(cdf.cpu(), byte_stream)
            decomp_attr = torchac.decode_int16_normalized_cdf(
                kit._convert_to_int_and_normalize(cdf, True).cpu(),
                byte_stream)
            decomp_attr = decomp_attr.view(1, -1, 1)
            context_attr_base = torch.cat((context_attr_base, decomp_attr), dim=1)
            i+=1
        decompressed_pc = torch.cat((batch_x_geo, context_attr_base), dim=-1)
        torch.cuda.synchronize()
        dec_times.append(time.time()-TIME_STAMP)
        decompressed_path = os.path.join(args.decompressed_path, fname+'.bin.ply')
        kit.save_point_cloud_reflactance(decompressed_pc[0].detach().cpu().numpy(), path=decompressed_path)


print('Max GPU Memory:', round(torch.cuda.max_memory_allocated(device=None)/1024/1024, 3), 'MB')
print('ave dec time:', round(np.array(dec_times).mean(), 3), 's')
