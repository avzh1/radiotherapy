#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --job-name=MedSAM
#SBATCH --mem=200GB
#SBATCH --gres=gpu:4
#SBATCH --output=logs/mgpus_%x-%j.out
#SBATCH --error=logs/mgpus_%x-%j.err
#SBATCH --time=20-00:00:00
#SBATCH -p gpushigh
#SBATCH --ntasks=1

# Options before:
# --nodes=5
# --ntasks=5
# --cpus-per-task=24
# --job-name=n-5nodes
# --mem=200GB
# --gres=gpu:4
# --partition=a100
# --output=logs/mgpus_%x-%j.out
# --error=logs/mgpus_%x-%j.err
# --time=20-00:00:00
# --exclude=gpu101,gpu113

# Get the directory of the script
SOURCE_DIR=$(git rev-parse --show-toplevel)

# Load the virtual environment
source ${SOURCE_DIR}/.venv/bin/activate
source ${SOURCE_DIR}/data/data_vars.sh

set -x -e

# log the sbatch environment
echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR

# Training setup
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE

## Master node setup
MAIN_HOST=`hostname -s`
export MASTER_ADDR=$MAIN_HOST

# Get a free port using python
export MASTER_PORT=$(python - <<EOF
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

export NNODES=$SLURM_NNODES
#NODE_RANK=$SLURM_PROCID ## do i need this?
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES)) # M nodes x N GPUs

echo "nnodes: ${NNODES}"

## Vector's cluster doesn't support infinite bandwidth
## but gloo backend would automatically use inifinite bandwidth if not disable
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

export NCCL_DEBUG=INFO

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
echo SLURM_NTASKS=$SLURM_NTASKS

echo pwd
cd $SOURCE_DIR/models/MedSAM/
echo pwd

for (( i=0; i<$SLURM_NTASKS; ++i ))
do
# /opt/slurm/bin/srun
    /usr/bin/srun -lN1 --mem=200G --gres=gpu:4 -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $i bash -c \
    "python train_multi_gpus.py \
        -task_name MedSAM-ViT-B-20GPUs \
        -work_dir ./work_dir \
        -batch_size 8 \
        -num_workers 8 \
        --world_size ${WORLD_SIZE} \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank ${i} \
        --init_method tcp://${MASTER_ADDR}:${MASTER_PORT}" >> ./logs/log_for_${SLURM_JOB_ID}_node_${i}.log 2>&1 &
done
wait ## Wait for the tasks on nodes to finish

echo "END TIME: $(date)"
