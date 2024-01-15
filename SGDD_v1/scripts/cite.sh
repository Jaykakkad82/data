
for s in 1000 120 85 15 1
do

    for r in 0.5 0.25 0.125
    do
        python train_sgdd_transduct.py --dataset citeseer --r=${r}  --seed=${s} --epoch=600 --save=0 --opt_scale=1e-9
        # python train_SGDD.py --dataset "citeseer"  --nlayers=2 --beta 0.1 --r=${r} --gpu_id=0 --seed=${s} --save=1
    done
done


