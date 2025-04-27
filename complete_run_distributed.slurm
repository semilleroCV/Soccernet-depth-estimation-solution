#!/bin/bash --login

hostname
nvidia-smi
pwd

echo "Loading Anaconda..."
conda activate soccernet

module purge
wandb login

#export NCCL_DEBUG=INFO

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
#master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
master_ip=$(hostname -I | cut -d' ' -f1)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${master_port}"

# Set environment variables
export OMP_NUM_THREADS=1

torchrun --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_addr=${master_ip} --master_port=${master_port} train.py \
  --data_dir soccernet_data \
  --crop_size 1078,1918 \
  --sport foot \
  --encoder vitl \
  --loss_fn ssigm \
  --precision float32 \
  --seed 2673 \
  --batch_size 2 \
  --epochs 7 \
  --learning_rate 5e-6 \
  --weight_decay 1e-3 \
  --optimizer adamw \
  --scheduler constant \
  --checkpoint_dir checkpoints/complete4 \
  --save_ckpt_freq 1 \
  --log_freq 200 \
  --val_freq 3 \
  --load_head \
  --wandb \
  --mask \
  "$@"