python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.005  --r=0.03 --seed=15 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 
python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.005  --r=0.03 --seed=85 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 
python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.005  --r=0.03 --seed=120 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 
python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.005  --r=0.03 --seed=1000 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 

python train_gcond_induct.py --dataset flickr --nlayers=2 --lr_feat=5e-3 --lr_adj=5e-3 --r=0.03  --sgc=0 --seed=1 --dis=mse --gpu_id=0 --one_step=1  --epochs=1000 --save=1
python train_gcond_induct.py --dataset flickr --nlayers=2 --lr_feat=5e-3 --lr_adj=5e-3 --r=0.03  --sgc=0 --seed=15 --dis=mse --gpu_id=0 --one_step=1  --epochs=1000 --save=1
python train_gcond_induct.py --dataset flickr --nlayers=2 --lr_feat=5e-3 --lr_adj=5e-3 --r=0.03  --sgc=0 --seed=85 --dis=mse --gpu_id=0 --one_step=1  --epochs=1000 --save=1
python train_gcond_induct.py --dataset flickr --nlayers=2 --lr_feat=5e-3 --lr_adj=5e-3 --r=0.03  --sgc=0 --seed=120 --dis=mse --gpu_id=0 --one_step=1  --epochs=1000 --save=1
python train_gcond_induct.py --dataset flickr --nlayers=2 --lr_feat=5e-3 --lr_adj=5e-3 --r=0.03  --sgc=0 --seed=1000 --dis=mse --gpu_id=0 --one_step=1  --epochs=1000 --save=1
