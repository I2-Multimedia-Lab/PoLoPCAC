# create an anaconda environment
# with python3.10, pytorch2.0.1 (CUDA 11.7), pytorch3d, and other dependencies
# tested on Ubuntu 20.04 and Debian GNU/Linux 10 in April 2024

# create environment
conda create -n polopcac python=3.10
conda activate polopcac

# install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# install torchac and others
pip install torchac
pip install ninja

pip install pandas matplotlib plyfile pyntcloud
