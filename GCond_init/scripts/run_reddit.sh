# for r in 0.002 0.0075 0.01
# do
for s in 1000 120 85 15 1
do

    for r in 0.001 0.005 0.0005
    do
        python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=${r} --seed=${s} --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 --inittype='herding'
    done
done
