# Cora: 0.25/0.5/1
python train_sgdd_transduct.py --dataset cora --r=0.5  --seed=1 --epoch=600 --save=0 --opt_scale=1e-11

# Citeseer: 0.25/0.5/1
python train_sgdd_transduct.py --dataset citeseer --r=0.5  --seed=1 --epoch=600 --save=0 --opt_scale=1e-9

# Ogbn-arxiv: 0.001/0.005/0.01
python train_gcond_transduct.py --dataset ogbn-arxiv --lr_feat=0.01 --r=0.005  --inner=3  --epochs=500  --save=0 --opt_scale=1e-12