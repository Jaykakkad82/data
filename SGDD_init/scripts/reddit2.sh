for s in 1000 120 85 15 1
do
    for r in 0.001
    do
        python train_SGDD.py --dataset "reddit"  --nlayers=2 --beta 0.1 --r=${r} --gpu_id=0 --seed=${s} --save=1
    done

    # for r in 0.001 0.005 0.01
    # do
    #     python train_SGDD.py --dataset "flickr"  --nlayers=2 --beta 0.1 --r=${r} --gpu_id=0 --seed=${s} --save=1
    # done
done

python train_SGDD.py --dataset "reddit"  --nlayers=2 --beta 0.1 --r=0.001 --gpu_id=0 --seed=15 --save=1
python train_SGDD.py --dataset "reddit"  --nlayers=2 --beta 0.1 --r=0.001 --gpu_id=0 --seed=85 --save=1
python train_SGDD.py --dataset "reddit"  --nlayers=2 --beta 0.1 --r=0.005 --gpu_id=0 --seed=120 --save=1
python train_SGDD.py --dataset "reddit"  --nlayers=2 --beta 0.1 --r=0.005 --gpu_id=0 --seed=85 --save=1
python train_SGDD.py --dataset "reddit"  --nlayers=2 --beta 0.1 --r=0.005 --gpu_id=0 --seed=15 --save=1
python train_SGDD.py --dataset "reddit"  --nlayers=2 --beta 0.1 --r=0.005 --gpu_id=0 --seed=1 --save=1
python train_SGDD.py --dataset "reddit"  --nlayers=2 --beta 0.1 --r=0.0005 --gpu_id=0 --seed=1 --save=1