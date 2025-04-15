export PYTHONPATH=$PYTHONPATH:./

########### stage 0: static image ###########
# 's0_batch_size': 24, 's0_iterations': 150000, 's0_finetune': 0, 's0_steps': [100000], 's0_lr': 5e-05, 's0_num_ref_frames': 2, 's0_num_frames': 3, 's0_start_warm': 20000, 's0_end_warm': 70000,

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --master_port 25473 \
--nproc_per_node=6 \
jointformer/train.py \
--exp_id stage0_static \
--stage 0 \
--s0_batch_size 24 --s0_iterations 150000 --s0_finetune 0 --s0_steps 100000 --s0_lr 5e-5 --s0_num_ref_frames 2 --s0_num_frames 3 --s0_start_warm 20000 --s0_end_warm 70000

########### stage 1: BL30K dataset ###########
# 's1_batch_size': 12, 's1_iterations': 300000, 's1_finetune': 0, 's1_steps': [200000], 's1_lr': 5e-05, 's1_num_ref_frames': 2, 's1_num_frames': 4, 's1_start_warm': 20000, 's1_end_warm': 70000,

s0_model_path=''

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --master_port 25473 \
--nproc_per_node=6 \
jointformer/train.py \
--exp_id stage1_BL30K_loads0 \
--stage 1 \
--s1_batch_size 12 --s1_iterations 300000 --s1_finetune 0 --s1_steps 200000 --s1_lr 5e-5 --s1_num_ref_frames 2 --s1_num_frames 4 --s1_start_warm 20000 --s1_end_warm 70000 \
--load_network $s0_model_path

########### stage 2: DAVIS 2017 & YoutubeVOS 2019 ###########
# 's2_batch_size': 12, 's2_iterations': 150000, 's2_finetune': 10000, 's2_steps': [100000], 's2_lr': 5e-05, 's2_num_ref_frames': 2, 's2_num_frames': 4, 's2_start_warm': 20000, 's2_end_warm': 70000

s01_model_path=''

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --master_port 25473 \
--nproc_per_node=6 \
jointformer/train.py \
--exp_id stage2_DAVISYTB_loads01 \
--stage 2 \
--s2_batch_size 12 --s2_iterations 150000 --s2_finetune 10000 --s2_steps 100000 --s2_lr 5e-5 --s2_num_ref_frames 2 --s2_num_frames 4 --s2_start_warm 20000 --s2_end_warm 70000 \
--load_network $s1_model_path

########### stage 4: VOST ###########
# 's4_batch_size': 12, 's4_iterations': 30000, 's4_finetune': 10000, 's4_steps': [20000], 's4_lr': 5e-05, 's4_num_ref_frames': 2, 's4_num_frames': 4, 's4_start_warm': 4000, 's4_end_warm': 13000,

s02_model_path=''

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --master_port 25473 \
--nproc_per_node=6 \
jointformer/train.py \
--exp_id stage4_VOST_loads02 \
--stage 4 \
--s4_batch_size 12 --s4_iterations 30000 --s4_finetune 10000 --s4_steps 20000 --s4_lr 5e-5 --s4_num_ref_frames 2 --s4_num_frames 4 --s4_start_warm 4000 --s4_end_warm 13000 \
--load_network $s02_model_path

########### stage 5: MOSE ###########
# 's5_batch_size': 12, 's5_iterations': 150000, 's5_finetune': 10000, 's5_steps': [100000], 's5_lr': 5e-05, 's5_num_ref_frames': 2, 's5_num_frames': 4, 's5_start_warm': 20000, 's5_end_warm': 70000,

s01_model_path=''

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --master_port 25473 \
--nproc_per_node=6 \
jointformer/train.py \
--exp_id stage5_MOSE_loads01 \
--stage 5 \
--s5_batch_size 12 --s5_iterations 150000 --s5_finetune 10000 --s5_steps 100000 --s5_lr 5e-5 --s5_num_ref_frames 2 --s5_num_frames 4 --s5_start_warm 20000 --s5_end_warm 70000 \
--load_network $s01_model_path

########### stage 6: LVOS ###########
# 's6_batch_size': 12, 's6_iterations': 5000, 's6_finetune': 5000, 's6_steps': [3000], 's6_lr': 5e-05, 's6_num_ref_frames': 2, 's6_num_frames': 4, 's6_start_warm': 1000, 's6_end_warm': 2500

s012_model_path=''

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --master_port 25473 \
--nproc_per_node=6 \
jointformer/train.py \
--exp_id stage5_LVOS_loads012 \
--stage 6 \
--s6_batch_size 12 --s6_iterations 5000 --s6_finetune 5000 --s6_steps 3000 --s6_lr 5e-5 --s6_num_ref_frames 2 --s6_num_frames 4 --s6_start_warm 1000 --s6_end_warm 2500 \
--load_network $s012_model_path

########### stage 7: VISOR ###########
# 's7_batch_size': 12, 's7_iterations': 150000, 's7_finetune': 10000, 's7_steps': [100000], 's7_lr': 5e-05, 's7_num_ref_frames': 2, 's7_num_frames': 4, 's7_start_warm': 20000, 's7_end_warm': 70000

s0_model_path=''

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --master_port 25473 \
--nproc_per_node=6 \
jointformer/train.py \
--exp_id stage5_VISOR_loads0 \
--stage 7 \
--s7_batch_size 12 --s7_iterations 150000 --s7_finetune 10000 --s7_steps 100000 --s7_lr 5e-5 --s7_num_ref_frames 2 --s7_num_frames 4 --s7_start_warm 20000 --s7_end_warm 70000 \
--load_network $s0_model_path