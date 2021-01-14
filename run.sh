#!/bin/bash

# Job details
TIME=05:59  # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=1  # GPUs per node
GPU_MODEL=GeForceGTX1080Ti #GeForceGTX1080Ti  # Choices: GeForceGTX1080,GeForceGTX1080Ti,GeForceRTX2080Ti,TeslaV100_SXM2_32GB
NUM_CPUS=1  # Number of cores (default: 1)
CPU_RAM=16384  # RAM for each core (default: 1024)
#OUTFILE=lsf.oJOBID  # default: lsf.oJOBID

# Load modules

module load gcc/6.3.0 python_gpu/3.7.4 hdf5/1.10.1

# Submit job
for arch in soft hard softinputfeed largesoftinputfeed approxihard \
approxihardinputfeed hmm hmmfull transformer universaltransformer \
tagtransformer taguniversaltransformer
do
  for train_dir in 1000exp1
  do
    for experiment in {0..99}
    do
      # Submit job
      bsub -W $TIME \
           -n $NUM_CPUS \
           -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" \
           -R "select[gpu_model0==${GPU_MODEL}]" \
           -R "select[gpu_mtotal0>=30000]" \
           "source ~/.bashrc; \
           conda activate precedent; \
           python src/train.py --dataset scan --train ${train_dir}/${experiment}/tasks_train_simple.txt --dev ${train_dir}/${experiment}/tasks_train_simple.txt --test ${train_dir}/${experiment}/tasks_test_simple.txt  --model model/${train_dir}/${arch}/${experiment} --embed_dim 100 \
           --src_hs 200 \
           --trg_hs 200 \
           --dropout 0.5 \
           --src_layer 2 \
           --trg_layer 2 \
           --max_norm 5 \
           --shuffle \
           --arch ${arch} \
           --gpuid 0 \
           --estop 1e-8 \
           --epochs 50 \
           --bs 512 \
           --cleanup_anyway \
           --total_eval 1 \
           --max_steps 100000"
    done
  done
done

