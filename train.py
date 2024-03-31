import os
import math
from glob import glob
import datetime

import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

import kit
from net import Network

torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)

TRAIN_GLOB = '/root/autodl-tmp/ShapeNet/ShapeNet_pc_01_2048p_colorful/train/train/*.ply'

BATCH_SIZE = 8
MAX_STEPS = 170000
LEARNING_RATE = 0.0005
LR_DECAY = 0.1
LR_DECAY_STEPS = 50000

MODEL_SAVE_FOLDER = f'./model/ycocg_3090trained/'

# CREATE MODEL SAVE PATH
if not os.path.exists(MODEL_SAVE_FOLDER):
    os.makedirs(MODEL_SAVE_FOLDER)

files = np.array(glob(TRAIN_GLOB, recursive=True))
np.random.shuffle(files)
files = files[:]
points = kit.read_point_clouds_ycocg(files)

# points = torch.tensor(points)
# print(f'Point train samples: {points.shape}, corrdinate range: [{points.min()}, {points.max()}]')

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset = points,
    batch_size = BATCH_SIZE,
    shuffle = True,
)

ae = Network(local_region=8, granularity=2**14, init_ratio=128, expand_ratio=2).cuda().train()
optimizer = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE)

bpps, losses = [], []
global_step = 0

for epoch in range(1, 9999):
    print(datetime.datetime.now())
    for step, (batch_x) in enumerate(loader):
        # batch_x = batch_x[:, :2048, :]
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
        if global_step % LR_DECAY_STEPS == 0:
            LEARNING_RATE = LEARNING_RATE * LR_DECAY
            for g in optimizer.param_groups:
                g['lr'] = LEARNING_RATE
            print(f'Learning rate decay triggered at step {global_step}, LR is setting to{LEARNING_RATE}.')

        # SAVE AND VIEW
        if global_step % 500 == 0:
            loss_value = loss.item()
            # SAVE MODEL
            torch.save(ae.state_dict(), MODEL_SAVE_FOLDER + f'ckpt.pt')
            
            # for i in range(1):
            #     kit.save_point_cloud(pred_batch_x[0].clip(0, 1).detach().cpu().numpy(), 
            #     f'./data/viewing_train/Step{global_step}_xhat_loss_{round(loss.item(), 5)}.ply', save_color=True)

            #     kit.save_point_cloud(batch_x[0].clip(0, 1).detach().cpu().numpy(), 
            #     f'./data/viewing_train/Step{global_step}_x_loss_{round(loss.item(), 5)}.ply', save_color=True)
        
        if global_step >= MAX_STEPS:
            break

    if global_step >= MAX_STEPS:
        break
