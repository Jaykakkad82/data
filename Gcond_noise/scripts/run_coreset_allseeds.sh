
for n in 0.1 0.15 0.2
do
    for s in 1 15 85 120 1000
    do
        for r in 0.5 0.25 0.125
        do     
            python train_coreset.py --dataset cora --r=${r}  --method=herding --seed=${s} --noise=${n}
            python train_coreset.py --dataset cora --r=${r}  --method=random  --seed=${s} --noise=${n}
            python train_coreset.py --dataset cora --r=${r}  --method=kcenter  --seed=${s} --noise=${n}
            python train_coreset.py --dataset cora --r=${r}  --method=kmeans  --seed=${s} --noise=${n}

        done

        for r in 0.5 0.25 0.125
        do
            python train_coreset.py --dataset citeseer --r=${r}  --method=herding --seed=${s} --noise=${n}
            python train_coreset.py --dataset citeseer --r=${r}  --method=random --seed=${s} --noise=${n}
            python train_coreset.py --dataset citeseer --r=${r}  --method=kcenter --seed=${s} --noise=${n}
            python train_coreset.py --dataset citeseer --r=${r}  --method=kmeans --seed=${s} --noise=${n}
        done
    done
done

    # for r in 0.001 0.005 0.01 0.03 0.06 0.08
    # do
    #     python train_coreset_induct.py --dataset flickr --r=${r}  --method=herding --seed=${s}
    #     python train_coreset_induct.py --dataset flickr --r=${r}  --method=random --seed=${s}
    #     python train_coreset_induct.py --dataset flickr --r=${r}  --method=kcenter --seed=${s}
    #     python train_coreset_induct.py --dataset flickr --r=${r}  --method=kmeans --seed=${s}
    # done

    # for r in 0.001 0.005 0.0005 0.075 0.01 0.02
    # do
    #     python train_coreset_induct.py --dataset reddit --r=${r}  --method=herding --seed=${s}
    #     python train_coreset_induct.py --dataset reddit --r=${r}  --method=random --seed=${s}
    #     python train_coreset_induct.py --dataset reddit --r=${r}  --method=kcenter --seed=${s}
    #     python train_coreset_induct.py --dataset reddit --r=${r}  --method=kmeans --seed=${s}
    # done
