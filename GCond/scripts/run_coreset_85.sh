
for r in 0.5 0.25 0.125
do     
    python train_coreset.py --dataset cora --r=${r}  --method=herding --seed=1000
    python train_coreset.py --dataset cora --r=${r}  --method=random  --seed=1000
    python train_coreset.py --dataset cora --r=${r}  --method=kcenter  --seed=1000

done

for r in 0.5 0.25 0.125
do
    python train_coreset.py --dataset citeseer --r=${r}  --method=herding --seed=1000
    python train_coreset.py --dataset citeseer --r=${r}  --method=random --seed=1000
    python train_coreset.py --dataset citeseer --r=${r}  --method=kcenter --seed=1000
done 
