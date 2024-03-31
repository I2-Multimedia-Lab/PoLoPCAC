
# Guidance for PoLoPCAC

## Environment

The environment we use is as follows：

Python 3.10.10

Pytorch 2.0.0 with CUDA 11.7

Pytorch3d 0.7.3

Torchac 0.9.3

## Data

Example point clouds are saved in ``./data/``.

## Compression

```
python ./compress.py  './model/ckpt.pt'  './data/2k-ShapeNet/*.ply'  './data/shapenet_compressed'
```

## Decompression

```
python ./decompress.py  './model/ckpt.pt'  './data/shapenet_compressed' './data/shapenet_decompressed'
```
