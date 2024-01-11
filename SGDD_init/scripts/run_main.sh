# 'cora', 'citeseer', 'pubmed', 'yelpchi', 'amazon', "sbm"

# python train_SGDD.py --dataset "citeseer"  --nlayers=2 --beta 0.1 --r=0.25 --gpu_id=0
# python train_SGDD.py --dataset 'pubmed' --nlayers=2 --beta 0.1 --r=0.5 --gpu_id=0
# python train_SGDD.py --dataset 'yelpchi'  --nlayers=2 --beta 0.1 --r=0.001 --gpu_id=0
# python train_SGDD.py --dataset 'amazon'  --nlayers=2 --beta 0.1 --r=0.002 --gpu_id=0
# python train_SGDD.py --dataset "sbm"  --nlayers=2 --beta 0.1 --r=0.25 --gpu_id=0

# python train_SGDD.py --dataset "cora"  --nlayers=2 --beta 0.1 --r=0.25 --gpu_id=0 --inittype='kcenter'

for s in 1000 120 85 15
do
    for r in 0.25 0.5
    do
        python train_SGDD.py --dataset "cora"  --nlayers=2 --beta 0.1 --r=${r} --gpu_id=0 --seed=${s} --inittype='kcenter'
        python train_SGDD.py --dataset "cora"  --nlayers=2 --beta 0.1 --r=${r} --gpu_id=0 --seed=${s} --inittype='herding'
    done

    for r in 0.25 0.5
    do
        python train_SGDD.py --dataset "citeseer"  --nlayers=2 --beta 0.1 --r=${r} --gpu_id=0 --seed=${s} --inittype='kcenter'
        python train_SGDD.py --dataset "citeseer"  --nlayers=2 --beta 0.1 --r=${r} --gpu_id=0 --seed=${s} --inittype='herding'
        #python train_gcond_transduct.py --dataset citeseer --nlayers=2 --lr_feat=1e-3 --lr_adj=1e-3 --r=${r}  --sgc=0 --dis=mse --one_step=1  --epochs=3000 --save=1 --seed=${s}
    done

done
