

for r in 0.001 0.005 0.0005
do
    python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=${r} --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 
done