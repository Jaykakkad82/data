for s in 1000 120 85 15
do
    for r in 0.25 0.5
    do
    python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=${r}  --seed=${s} --epoch=600 --save=1 --inittype='herding'
    python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=${r}  --seed=${s} --epoch=600 --save=1 --inittype='kcenter'
    #python train_gcond_transduct.py --dataset cora --nlayers=2 --lr_feat=1e-3 --gpu_id=0 --lr_adj=1e-3 --r=${r}  --sgc=0  --seed=${s} --dis=mse --one_step=1  --epochs=5000 --save=1
    done

    for r in 0.25 0.5
    do
        python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=${r}  --seed=${s} --epoch=600 --save=1 --inittype='herding'
        python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=${r}  --seed=${s} --epoch=600 --save=1 --inittype='kcenter'
        #python train_gcond_transduct.py --dataset citeseer --nlayers=2 --lr_feat=1e-3 --lr_adj=1e-3 --r=${r}  --sgc=0 --dis=mse --one_step=1  --epochs=3000 --save=1 --seed=${s}
    done

    # for r in 0.001 0.005 0.01
    # do
    #     #python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.005  --r=${r} --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 
    #     # python train_gcond_induct.py --dataset flickr --nlayers=2 --lr_feat=5e-3 --lr_adj=5e-3 --r=${r}  --sgc=0 --seed=${s} --dis=mse --gpu_id=0 --one_step=1  --epochs=1000 --save=1

    # done

    # for r in 0.001 0.005 0.0005
    # do
    #     #python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=${r} --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 
    #     # python train_gcond_induct.py --dataset reddit --nlayers=2 --lr_feat=5e-3 --lr_adj=5e-3 --r=${r}  --sgc=0 --seed=${s} --dis=mse --gpu_id=0 --one_step=1  --epochs=1000 --save=1

    # done
done
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=600 --save=1 --inittype='herding'

