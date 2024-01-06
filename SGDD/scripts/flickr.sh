for s in 1000 120 85 15 1
do
    # for r in 0.0001 0.0005 0.005
    # do
    #     python train_SGDD.py --dataset "reddit"  --nlayers=2 --beta 0.1 --r=${r} --gpu_id=0 --seed=${s} --save=1
    # done

    for r in 0.001 0.005 0.01
    do
        python train_SGDD.py --dataset "flickr"  --nlayers=2 --beta 0.1 --r=${r} --gpu_id=0 --seed=${s} --save=1
    done
done