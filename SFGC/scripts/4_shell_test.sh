#@citeseer-r025
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.25 \
--test_lr_model=0.003 --test_wd=0.005 --test_model_iters 1000 --nruns 1  --test_dropout=0.3 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_0.25-20221118-154807-807563'

#@citeseer-r05
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.5 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 1000 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_0.5-20221206-032653-877857'

#@citeseer-r1
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 1 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 1000 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_1.0-20221108-104444-053909'

#@cora-r025
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.25 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/cora-reduce_0.25-20221102-142040-290477'

#@cora-r05
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset cora \
--reduction_rate 0.5 \
--test_lr_model=0.001 --test_wd=0.001 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/cora-reduce_0.5-20221031-140919-044426'

#@cora-r1
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset cora \
--reduction_rate 1 \
--test_lr_model=0.01 --test_wd=0.0005 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/cora-reduce_1.0-20221101-055335-943669'


#@ogbn-r0001
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset ogbn-arxiv \
--reduction_rate 0.001 \
--test_lr_model=0.001 --test_wd=0.005 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/ogbn-arxiv-reduce_0.001-20230109-071146-501074'

#@ogbn-r0005
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset ogbn-arxiv \
--reduction_rate 0.005 \
--test_lr_model=0.001 --test_wd=0.005 --test_model_iters 1000 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/ogbn-arxiv-reduce_0.005-20230110-092420-750673'

#@ogbn-r001
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset ogbn-arxiv \
--reduction_rate 0.005 \
--test_lr_model=0.01 --test_wd=0.005 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/ogbn-arxiv-reduce_0.01-20221103-114311-982859'

#@flickr-r0001
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.001 \
--test_lr_model=0.001 --test_wd=0.5 --test_model_iters 200 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.001-20221123-073712-860439'

#@flickr-r0005
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.005 \
--test_lr_model=0.0005 --test_wd=0.07 --test_model_iters 200 --nruns 1  --test_dropout=0.6 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.005-20230126-190052-005881'

#@flickr-r001
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset flickr \
--reduction_rate 0.01 \
--test_lr_model=0.001 --test_wd=0.5 --test_model_iters 200 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/flickr-reduce_0.001-20221123-073712-860439'

#@reddit-r00005
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset reddit \
--reduction_rate 0.0005 \
--test_lr_model=0.005 --test_wd=0.005 --test_model_iters 200 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/reddit-reduce_0.0005-20221217-124831-730479'

#@reddit-r0001
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset reddit \
--reduction_rate 0.001 \
--test_lr_model=0.01 --test_wd=0.005 --test_model_iters 300 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/reddit-reduce_0.001-20221219-050913-075170'

#@reddit-r0002
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset reddit \
--reduction_rate 0.002 \
--test_lr_model=? --test_wd=? --test_model_iters ? --nruns 1  --test_dropout=? \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path=?

#@reddit-r0005
CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset reddit \
--reduction_rate 0.005 \
--test_lr_model=0.01 --test_wd=0.003 --test_model_iters 300 --nruns 1  --test_dropout=0.3 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/reddit-reduce_0.005-20230118-121157-186483'
