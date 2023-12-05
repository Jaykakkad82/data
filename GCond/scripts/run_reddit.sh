# for r in 0.002 0.0075 0.01
# do
# for s in 1 15 120 1000 85
# do
#     python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=${r} --seed=${s} --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 
# done
# done

for s in 1 15 120 1000 85
do
    python train_coreset_induct.py --dataset reddit --r=0.002  --method=herding --seed=${s}
    python train_coreset_induct.py --dataset reddit --r=0.002  --method=random --seed=${s}
    python train_coreset_induct.py --dataset reddit --r=0.002 --method=kcenter --seed=${s}
    python train_coreset_induct.py --dataset reddit --r=0.002  --method=kmeans --seed=${s}
done