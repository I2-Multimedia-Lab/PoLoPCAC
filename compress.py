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
    prog='compress.py',
    description='Compress Point Cloud Attributes.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ckpt', required=True, help='Trained ckeckpoint file.')
parser.add_argument('--input_glob', required=True, help='Point clouds glob pattern to be compressed.')
parser.add_argument('--compressed_path', required=True, help='Compressed file saving directory.')

parser.add_argument('--local_region', type=int, help='Neighbooring scope for context windows (i.e., K).', default=8)
parser.add_argument('--granularity', type=int, help='Upper limit for each group (i.e., s*).', default=2**14)
parser.add_argument('--init_ratio', type=int, help='The ratio for size of the very first group (i.e., alpha).', default=128)
parser.add_argument('--expand_ratio', type=int, help='Expand ratio (i.e., r)', default=2)
parser.add_argument('--prg_seed', type=int, help='Pseudorandom seed for PRG.', default=2147483647)

args = parser.parse_args()


if not os.path.exists(args.compressed_path):
    os.makedirs(args.compressed_path)

files = np.array(glob(args.input_glob, recursive=True))
np.random.shuffle(files)

net = Network(local_region=args.local_region, granularity=args.granularity, init_ratio=args.init_ratio, expand_ratio=args.expand_ratio)
net.load_state_dict(torch.load(args.ckpt))
net = torch.compile(net, mode='max-autotune')
net.cuda().eval()

# warm up our model
# since the very first step of network is extremely slow...
_ = net.mu_sigma_pred(net.pt(torch.rand((1, 32, 8, 3)).cuda(), torch.rand((1, 32, 8, 3)).cuda()))
 
enc_times = []
fnames, bpps = [], []
with torch.no_grad():
    for f in tqdm(files):
        fname = os.path.split(f)[-1]

        pc = kit.read_point_cloud_ycocg(f)
        batch_x = torch.tensor(pc).unsqueeze(0)
        batch_x = batch_x.cuda()

        B, N, _ = batch_x.shape
        
        torch.cuda.synchronize()
        TIME_STAMP = time.time()

        #####################################
        # progressive random grouping

        g_cpu = torch.Generator()
        g_cpu.manual_seed(args.prg_seed)

        batch_x = batch_x[:, torch.randperm(batch_x.size()[1], generator=g_cpu), :]
        _, N, _ = batch_x.shape

        base_size = min(N//args.init_ratio, args.granularity)
        window_size = base_size

        context_ls, target_ls = [], []
        cursor = base_size
        
        while cursor<N:
            window_size = min(window_size*args.expand_ratio, args.granularity)
            context_ls.append(batch_x[:, :cursor, :])
            target_ls.append(batch_x[:, cursor:cursor+window_size, :])
            cursor += window_size

        #####################################
        # network infering

        total_ac_time = 0
        total_bits = 0
        for i in range(len(target_ls)):

            target_geo, target_attr = target_ls[i][:, :, :3].clone(), target_ls[i][:, :, 3:].clone()
            context_geo, context_attr =  context_ls[i][:, :, :3].clone(), context_ls[i][:, :, 3:].clone()

            # rescale input attributes to [0, 1] in GPU
            context_attr[:, :, 0] = context_attr[:, :, 0] / 255
            context_attr[:, :, 1:] = context_attr[:, :, 1:] / 511
            
            # context window gathering
            _, idx, context_grouped_geo = knn_points(target_geo, context_geo, K=net.local_region, return_nn=True)
            context_grouped_attr = knn_gather(context_attr, idx)

            # spatial normalization
            context_grouped_geo = context_grouped_geo - target_geo.view(B, -1, 1, 3)
            context_grouped_geo = kit.n_scale_ball(context_grouped_geo)
            
            # network
            feature = net.pt(context_grouped_geo, context_grouped_attr)
            mu_sigma = net.mu_sigma_pred(feature)
            mu, sigma = mu_sigma[:, :, :3]+0.5, torch.exp(mu_sigma[:, :, 3:])

            cdf = kit.get_cdf_ycocg(mu[0]*255, sigma[0]*32)
            target_feature = (target_attr[0]).to(torch.int16)

            # original torchac.encode_float_cdf is pretty slow
            # put _convert_to_int_and_normalize in GPU -> faster
            # byte_stream = torchac.encode_float_cdf(cdf.cpu(), target_feature.cpu(), check_input_bounds=True)
            byte_stream = torchac.encode_int16_normalized_cdf(
                kit._convert_to_int_and_normalize(cdf, True).cpu(),
                target_feature.cpu())

            # save current group to a file
            # concat to a single bitstream is also practicable
            comp_f = os.path.join(args.compressed_path, fname+f'.{i}.bin')
            with open(comp_f, 'wb') as fout:
                fout.write(byte_stream)

            # record bitrate of current group
            total_bits += kit.get_file_size_in_bits(comp_f)
        
        # save the first group directly using 8bit code
        comp_base_f = os.path.join(args.compressed_path, fname+'.c.bin')
        context_base = context_ls[0][0, :, 3:].detach().cpu().numpy().astype(np.int16)
        context_base = kit.transformYCoCgToRGB(8, context_base)

        torch.cuda.synchronize()
        enc_times.append(time.time() - TIME_STAMP)
        
        context_base.astype(np.uint8).tofile(comp_base_f)
        total_bits += kit.get_file_size_in_bits(comp_base_f)

        # save geometry (for decompression only)
        geo_f = os.path.join(args.compressed_path, fname+'.geo.bin')
        batch_x[:, :, :3].detach().cpu().numpy().astype(np.float32).tofile(geo_f)

        # record
        fnames.append(fname)
        bpps.append(np.round(total_bits/N, 3))

print('Max GPU Memory:', round(torch.cuda.max_memory_allocated()/1024/1024, 3), 'MB')
print(f'Done! Total {len(fnames)} \
      | color bpp: {round(np.array(bpps).mean(), 3)}\
      | ave enc time: {round(np.array(enc_times).mean(), 3)} s')
print('Params:', sum(p.numel() for p in net.parameters()), 
      'Trainable params:', sum(p.numel() for p in net.parameters() if p.requires_grad))
