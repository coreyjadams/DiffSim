#!/bin/bash -l
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:grand


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/NEXT/DiffSim
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`

NRANKS_PER_NODE=4


let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=128
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

# Set up software deps:
source /home/cadams/Polaris/NP-with-ML/JAX_QMC_2/setup-polaris.sh

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB


run_id=diffSim-Sup-weight-1-$LOCAL_BATCH_SIZE

CPU_AFFINITY=24-31:16-23:8-15:0-7
export OMP_NUM_THREADS=8


# mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind list:${CPU_AFFINITY} \
python bin/exec.py \
--config-name krypton_supervised \
run.id=${run_id} \
run.minibatch_size=${LOCAL_BATCH_SIZE} \
run.iterations=2000 \
run.image_iteration=50 \
run.distributed=True
