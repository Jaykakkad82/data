for n in 0.1 0.15 0.2
do
    for s in 1000 120 85
    do
        for r in 0.25 0.5
        do
        python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=${r}  --seed=${s} --epoch=600 --save=1 --noise=${n}
        #python train_gcond_transduct.py --dataset cora --nlayers=2 --lr_feat=1e-3 --gpu_id=0 --lr_adj=1e-3 --r=${r}  --sgc=0  --seed=${s} --dis=mse --one_step=1  --epochs=5000 --save=1
        done

        for r in 0.25 0.5
        do
            python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=${r}  --seed=${s} --epoch=600 --save=1 --noise=${n}
            #python train_gcond_transduct.py --dataset citeseer --nlayers=2 --lr_feat=1e-3 --lr_adj=1e-3 --r=${r}  --sgc=0 --dis=mse --one_step=1  --epochs=3000 --save=1 --seed=${s}
        done

    done
done