#!/bin/bash

python_script="output"

# Python code to initialize the storage class
python_code=$(cat <<EOF
from $python_script import datastorage

seed_list = [1, 15, 85, 120, 1000]
init_list = ["herding", "kcenter"]
compr_list = {"cora": [0.25, 0.5], "citeseer": [0.25, 0.5], "reddit": [0.001, 0.005, 0.01]}
data_list = ["cora", "citeseer", "reddit"]
method_list = ["gcond", "sgdd"]

out = datastorage(seed_list, init_list, compr_list, data_list, method_list)
print(out)
EOF
)

# Run the Python code and capture the output
export myouttable=$(python3 -c "$python_code")

# using two new arguments : inittype and outinstance
# Pass the class instance to different arguments
python train_gcond_transduct.py --inittype kcenter --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=600 --save=1
python train_gcond_transduct.py --inittype kcenter --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=600 --save=1
