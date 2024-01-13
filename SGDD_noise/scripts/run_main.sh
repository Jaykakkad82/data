# 'cora', 'citeseer', 'pubmed', 'yelpchi', 'amazon', "sbm"

# python train_SGDD.py --dataset "citeseer"  --nlayers=2 --beta 0.1 --r=0.25 --gpu_id=0
# python train_SGDD.py --dataset 'pubmed' --nlayers=2 --beta 0.1 --r=0.5 --gpu_id=0
python train_SGDD.py --dataset 'yelpchi'  --nlayers=2 --beta 0.1 --r=0.001 --gpu_id=0
python train_SGDD.py --dataset 'amazon'  --nlayers=2 --beta 0.1 --r=0.002 --gpu_id=0
# python train_SGDD.py --dataset "sbm"  --nlayers=2 --beta 0.1 --r=0.25 --gpu_id=0