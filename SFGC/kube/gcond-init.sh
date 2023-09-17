#! /bin/bash

apt update
apt install screen

conda create -n gcond_env python=3.8 -y

/opt/conda/envs/gcond_env/bin/pip install -r /data/requirements.txt
/opt/conda/envs/gcond_env/bin/pip install torch_sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.0+cu102.html
/opt/conda/envs/gcond_env/bin/pip install torch_scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.0+cu102.html
/opt/conda/envs/gcond_env/bin/pip install torch_cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.7.0+cu102.html
/opt/conda/envs/gcond_env/bin/pip install torch_spline_conv==1.2.1 -f https://data.pyg.org/whl/torch-1.7.0+cu102.html

/opt/conda/envs/gcond_env/bin/python -c "import torch, torch_geometric; print('torch: {}, cuda: {}, geometric: {}'.format(torch.__version__, torch.version.cuda, torch_geometric.__version__))" > /workspace/lib_ver.txt