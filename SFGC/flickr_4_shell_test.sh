#@flickr-r0001
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.001 \
--test_lr_model=0.001 --test_wd=0.5 --test_model_iters 200 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.001-20231207-104656-844477' \
--tr_seed=31 --seed=31

#@flickr-r0005
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.005 \
--test_lr_model=0.0005 --test_wd=0.07 --test_model_iters 200 --nruns 1  --test_dropout=0.6 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.005-20231207-104555-778530' \
--tr_seed=31 --seed=31

#@flickr-r001
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.01 \
--test_lr_model=0.001 --test_wd=0.5 --test_model_iters 200 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.01-20231207-104821-110106' \
--tr_seed=31 --seed=31