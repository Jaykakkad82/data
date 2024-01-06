#! /bin/bash

# conda create -n exp2_env python=3.8 -y

# /opt/conda/envs/exp2_env/bin/pip install -r /data/requirements_exp2.txt
# /opt/conda/envs/exp2_env/bin/pip install torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# /opt/conda/envs/exp2_env/bin/pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# /opt/conda/envs/exp2_env/bin/pip install torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# /opt/conda/envs/exp2_env/bin/pip install torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# /opt/conda/envs/exp2_env/bin/python -c "import torch, torch_geometric; print('torch: {}, cuda: {}, geometric: {}'.format(torch.__version__, torch.version.cuda, torch_geometric.__version__))" > /workspace/lib_ver.txt
