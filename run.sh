#!/bin/bash

# Job details
TIME=01:00  # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=1  # GPUs per node
GPU_MODEL=TeslaV100_SXM2_32GB #GeForceGTX1080Ti  # Choices: GeForceGTX1080,GeForceGTX1080Ti,GeForceRTX2080Ti,TeslaV100_SXM2_32GB
NUM_CPUS=1  # Number of cores (default: 1)
CPU_RAM=16384  # RAM for each core (default: 1024)
#OUTFILE=lsf.oJOBID  # default: lsf.oJOBID

# Load modules

module load gcc/6.3.0 python_gpu/3.7.4 hdf5/1.10.1

# Submit job
gpu=0
train_dir=SCAN/tasks_train_simple.txt
test_dir=SCAN/tasks_test_simple.txt
ckpt_dir=model/scan/scan
exp=speed
arch=soft

# Submit job
bsub -W $TIME \
     -n $NUM_CPUS \
     -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" \
     -R "select[gpu_model0==${GPU_MODEL}]" \
     -R "select[gpu_mtotal0>=30000]" \
     "source ~/.bashrc; \
     conda activate precedent; \
     python src/train.py --dataset scan --train ${train_dir} --dev ${train_dir} --test ${test_dir}  --model model/scan/speed \
     --embed_dim 100 \
     --src_hs 200 \
     --trg_hs 200 \
     --dropout 0.2 \
     --src_layer 2 \
     --trg_layer 1 \
     --max_norm 5 \
     --shuffle \
     --arch ${arch} \
     --gpuid 0 \
     --estop 1e-8 \
     --epochs 50 \
     --bs 256 \
     --cleanup_anyway \
     --total_eval 1"
