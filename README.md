# Structure-free Graph Condensation (SFGC): From Large-scale Graphs to Condensed Graph-free Data

This is the Pytorch implementation for "Structure-free Graph Condensation (SFGC): From Large-scale Graphs to Condensed Graph-free Data"

Paper can be found in: https://arxiv.org/pdf/2306.02664.pdf

The framework is:
![image](https://user-images.githubusercontent.com/61812981/221344391-11904a34-fc9c-479b-9f8d-02b72d2bf56b.png)


### Requirements
## Instructions

(1) Run to generate the buffer for keeping the model's training parameter distribution (training trajectory)

For examples:

```
CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 0.001 \
--teacher_epochs 800 --dataset citeseer --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' \
--num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0
```
Detailed parameters and scripts can be found in scripts/1_shell_buffer.sh


(2) Use the coreset method to initialize the synthesized small-scale graph node features

For examples:

```
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1 --lr 0.01 --lr_coreset 0.005 \
--weight_decay 5e-4 --wd_coreset 5e-4 --save 1 --method kcenter --reduction_rate 0.5

CUDA_VISIBLE_DEVICES=3 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.001 --load_npy ''
```
Detailed parameters and scripts can be found in scripts/2_shell_coreset.sh


(3) Distill under training trajectory and coreset initialization to generate sythensized small-scale structure-free graph data

For examples:
```
CUDA_VISIBLE_DEVICES=2 python distill_inductive_adj_identity.py --dataset flickr --device cuda:0 --lr_feat=0.005 --optimizer_con Adam \
--expert_epochs=${exp_e} --lr_student=0.1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=50 --syn_steps=${syn_e} \
--buffer_path './logs/Buffer/flickr-20221108-120636-260182' \
--coreset_init_path './logs/Coreset/flickr-reduce_0.01-20221109-101537-112764' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.01 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter 1
```
Detailed parameters and scripts can be found in scripts/3_shell_distill.sh

(4) Training with the small-scale structure free graph data and test on the large-scale graph test set:

For example:

```
CUDA_VISIBLE_DEVICES=2 python test_other_arcs.py --device cuda:0 --dataset flickr --reduction_rate 0.01 \
--test_lr_model 0.001 --test_wd 5e-2 --test_model_iters 200 --nruns 1 --seed 0 \
--best_ntk_acc 0 --best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path=${your_generated_data_path_from_step(3)}
```
Detailed parameters and scripts can be found in scripts/4_shell_test.sh


### Abalation Study:

A1: Cross-architecture test
#['GAT', 'MLP', 'APPNP', 'GraphSage', 'Cheby','GCN']

For example：

```
CUDA_VISIBLE_DEVICES=1 python test_other_arcs_notgcn.py --device cuda:0 --dataset citeseer --reduction_rate 0.5 \
--test_lr_model 0.01 --test_wd 7e-2 --test_model_iters 300 --nruns 1 --seed 0 \
--best_ntk_acc 0 --best_ntk_score 1 --test_opt_type Adam --test_model_type SGC \
--load_path='./logs/Distill/citeseer-reduce_0.5-20221206-032653-877857'

CUDA_VISIBLE_DEVICES=1 python test_other_arcs.py --device cuda:0 --dataset citeseer --reduction_rate 0.5 \
--test_lr_model 0.005 --test_wd 1e-3 --test_model_iters 300 --nruns 1 --seed 0 \
--best_ntk_acc 0 --best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_0.5-20221206-032653-877857'

CUDA_VISIBLE_DEVICES=1 python test_other_arcs_gat.py --device cuda:0 --dataset cora --reduction_rate 0.5 \
--test_lr_model 0.001 --test_wd 5e-4 --nruns 1 --seed 0 \
--best_ntk_acc 0 --best_ntk_score 1 --test_opt_type Adam --test_model_type GAT \
--load_path='./logs/Distill/cora-reduce_0.5-20221031-140919-044426'
```

A2: Structure vs. Structure-free test

For example:

```
CUDA_VISIBLE_DEVICES=2 python distill_transduct_continue.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=0.2 --optimizer_con Adam \
--expert_epochs=1800 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=600 \
--buffer_path './logs/Buffer/used/ogbn-arxiv-20220925-223423-976629' \
--coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.001-20221025-212029-649373' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg 0.1 --eval_interval 10 --ITER 200 --samp_iter 1 --samp_num_per_class 50
```
Detailed parameters and scripts can be found in scripts/strcuture_vs_structure-free-{dataset}.sh


