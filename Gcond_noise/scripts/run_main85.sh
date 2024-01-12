# for r in 0.25 0.5 1
# do
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=${r}  --seed=85 --epoch=600 --save=1
# done

# for r in 0.25 0.5 1
# do
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=${r}  --seed=85 --epoch=600 --save=1
# done


# for r in 0.001 0.005 0.01
# do
# python train_gcond_transduct.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=3  --lr_adj=0.01 --r=${r}  --seed=120 --inner=3  --epochs=1000  --save=0
# done

#python train_gcond_transduct.py --dataset cora --nlayers=2 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.5 

# for r in 0.03 0.06 0.08
# do
#     python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.005  --r=${r} --seed=1000 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 
# done

# for r in 0.075 0.01 0.02
# do
#     python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=${r} --seed=1000 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 
# done

python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.125  --seed=15 --epoch=600 --save=1
python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.125  --seed=85 --epoch=600 --save=1
python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.125  --seed=85 --epoch=600 --save=1
python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.005  --r=0.001 --seed=15 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1 
python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --seed=15 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1
python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --seed=120 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=1
