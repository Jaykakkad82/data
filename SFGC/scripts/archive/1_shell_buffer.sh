#@citeseer-buff
CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 0.001 \
--teacher_epochs 800 --dataset citeseer --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' \
--num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0

#@cora-buff
CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 0.4 \
--teacher_epochs 2500 --dataset cora --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' \
--num_experts=200 --wd_teacher 0 --mom_teacher 0 --optim SGD

#@ogbn-buff
CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 1 \
--teacher_epochs 2000 --dataset ogbn-arxiv --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' \
--num_experts=200 --wd_teacher 0 --mom_teacher 0 --optim SGD


#@flickr-buff
CUDA_VISIBLE_DEVICES=0 python buffer_inductive.py --device cuda:0 --lr_teacher 0.001 \
--teacher_epochs 1000 --dataset flickr --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' \
--num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0

#@reddit-buff
CUDA_VISIBLE_DEVICES=0 python buffer_inductive.py --device cuda:0 --lr_teacher 0.001 \
--teacher_epochs 1000 --dataset reddit --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' \
--num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0




