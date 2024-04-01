import os
import argparse
import datetime

import numpy as np
from glob import glob

import torch
import torch.utils.data as Data
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

import kit
from net import Network

torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)


parser = argparse.ArgumentParser(
    prog='train.py',
    description='Training from scratch.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--training_data', required=True, help='Training data (Glob pattern).')
parser.add_argument('--model_save_folder', required=True, help='Directory where to save trained models.')

parser.add_argument('--learning_rate', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--batch_size', type=int, help='Batch size.', default=8)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', type=int, help='Decays the learning rate every x steps.', default=50000)
parser.add_argument('--max_steps', type=int, help='Train up to this number of steps.', default=170000)

parser.add_argument('--local_region', type=int, help='Neighbooring scope for context windows (i.e., K).', default=8)
parser.add_argument('--granularity', type=int, help='Upper limit for each group (i.e., s*).', default=2**14)
parser.add_argument('--init_ratio', type=int, help='The ratio for size of the very first group (i.e., alpha).', default=128)
parser.add_argument('--expand_ratio', type=int, help='Expand ratio (i.e., r)', default=2)

args = parser.parse_args()

# CREATE MODEL SAVE PATH
if not os.path.exists(args.model_save_folder):
    os.makedirs(args.model_save_folder)

files = np.array(glob(args.training_data, recursive=True))
np.random.shuffle(files)
files = files[:]
points = kit.read_point_clouds_ycocg(files)

loader = Data.DataLoader(
    dataset = points,
    batch_size = args.batch_size,
    shuffle = True,
)

ae = Network(local_region=args.local_region, granularity=args.granularity, init_ratio=args.init_ratio, expand_ratio=args.expand_ratio).cuda().train()
optimizer = torch.optim.Adam(ae.parameters(), lr=args.learning_rate)

bpps, losses = [], []
global_step = 0

for epoch in range(1, 9999):
    print(datetime.datetime.now())
    for step, (batch_x) in enumerate(loader):
        B, N, _ = batch_x.shape
        batch_x = batch_x.cuda()

        optimizer.zero_grad()

        total_bits = ae(batch_x)
        bpp = total_bits / B / N
        loss = bpp

        loss.backward()

        optimizer.step()
        global_step += 1

        # PRINT
        losses.append(loss.item())
        bpps.append(bpp.item())

        if global_step % 500 == 0:
            print(f'Epoch:{epoch} | Step:{global_step} | bpp:{round(np.array(bpps).mean(), 5)} | Loss:{round(np.array(losses).mean(), 5)}')
            bpps, losses = [], []
        
         # LEARNING RATE DECAY
        if global_step % args.lr_decay_steps == 0:
            args.learning_rate = args.learning_rate * args.lr_decay
            for g in optimizer.param_groups:
                g['lr'] = args.learning_rate
            print(f'Learning rate decay triggered at step {global_step}, LR is setting to{args.learning_rate}.')

        # SAVE MODEL
        if global_step % 500 == 0:
            torch.save(ae.state_dict(), args.model_save_folder + f'ckpt.pt')
        
        if global_step >= args.max_steps:
            break

    if global_step >= args.max_steps:
        break
