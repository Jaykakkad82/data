#@flickr-r0001
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.001 \
--test_lr_model=0.001 --test_wd=0.5 --test_model_iters 200 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.001-20240111-235643-843270' \
--tr_seed=61 --seed=61

#@flickr-r0005
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.005 \
--test_lr_model=0.0005 --test_wd=0.07 --test_model_iters 200 --nruns 1  --test_dropout=0.6 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.005-20240112-094815-746390' \
--tr_seed=61 --seed=61

#@flickr-r001
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.01 \
--test_lr_model=0.001 --test_wd=0.5 --test_model_iters 200 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.01-20240112-151101-970813' \
--tr_seed=61 --seed=61


#@flickr-r0001
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.001 \
--test_lr_model=0.001 --test_wd=0.5 --test_model_iters 200 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.001-20240112-003537-991925' \
--tr_seed=5 --seed=5

#@flickr-r0005
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.005 \
--test_lr_model=0.0005 --test_wd=0.07 --test_model_iters 200 --nruns 1  --test_dropout=0.6 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.005-20240112-042554-859658' \
--tr_seed=5 --seed=5

#@flickr-r001
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.01 \
--test_lr_model=0.001 --test_wd=0.5 --test_model_iters 200 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.01-20240112-082229-620134' \
--tr_seed=5 --seed=5