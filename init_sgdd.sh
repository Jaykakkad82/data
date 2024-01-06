#! /bin/bash

# conda create -n exp2_env python=3.8 -y
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch_geometric
pip install deeprobust==0.2.8

# pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
# pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

# /opt/conda/envs/exp2_env/bin/pip install -r /data/requirements_exp2.txt
# /opt/conda/envs/exp2_env/bin/pip install torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# /opt/conda/envs/exp2_env/bin/pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# /opt/conda/envs/exp2_env/bin/pip install torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# /opt/conda/envs/exp2_env/bin/pip install torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# /opt/conda/envs/exp2_env/bin/python -c "import torch, torch_geometric; print('torch: {}, cuda: {}, geometric: {}'.format(torch.__version__, torch.version.cuda, torch_geometric.__version__))" > /workspace/lib_ver.txt
