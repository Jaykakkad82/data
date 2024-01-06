#@reddit-r00005
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset reddit \
--reduction_rate 0.0005 \
--test_lr_model=0.005 --test_wd=0.005 --test_model_iters 200 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/reddit-reduce_0.0005-20240104-093156-953220' \
--tr_seed=31 --seed=31

# #@reddit-r0002
# CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset reddit \
# --reduction_rate 0.002 \
# --test_lr_model=? --test_wd=? --test_model_iters ? --nruns 1  --test_dropout=? \
# --best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
# --load_path=?

#@reddit-r0005
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset reddit \
--reduction_rate 0.005 \
--test_lr_model=0.01 --test_wd=0.003 --test_model_iters 300 --nruns 1  --test_dropout=0.3 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/reddit-reduce_0.005-20240104-093117-631207' \
--tr_seed=31 --seed=31

#@reddit-r0001
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset reddit \
--reduction_rate 0.001 \
--test_lr_model=0.01 --test_wd=0.005 --test_model_iters 300 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/reddit-reduce_0.001-20240104-093022-555378' \
--tr_seed=31 --seed=31
