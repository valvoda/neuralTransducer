#!/bin/bash

# Job details
TIME=24:00  # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=1  # GPUs per node
GPU_MODEL=TeslaV100_SXM2_32GB #GeForceGTX1080Ti  # Choices: GeForceGTX1080,GeForceGTX1080Ti,GeForceRTX2080Ti,TeslaV100_SXM2_32GB
NUM_CPUS=1  # Number of cores (default: 1)
CPU_RAM=16384  # RAM for each core (default: 1024)
OUTFILE=./logs/model3.oJOBID  # default: lsf.oJOBID

# Load modules

module load gcc/6.3.0 python_gpu/3.7.4 hdf5/1.10.1

# Submit job
for arch in soft
#hard softinputfeed largesoftinputfeed approxihard \
#approxihardinputfeed hmm hmmfull transformer universaltransformer \
#tagtransformer taguniversaltransformer
do
  for experiment in 2 3
  do
    for size in 50 100 300
    do
      # 10exp1_run 20exp1_run 30exp1_run 40exp1_run 60exp1_run 70exp1_run 80exp1_run 90exp1_run 50exp1_run 100exp1_run 200exp1_run 400exp1_run 600exp1_run 800exp1_run 1000exp1_run 200exp1_run 300exp1_run 500exp1_run 700exp1_run 900exp1_run
      # 50exp2_run 100exp2_run 200exp2_run 400exp2_run 600exp2_run 800exp2_run 1000exp2_run 200exp2_run 300exp2_run 500exp2_run 700exp2_run 900exp2_run
      for train_dir in 26exp1_run 30exp1_run 36exp1_run
      do
        # Submit job
        bsub -W $TIME \
             -n $NUM_CPUS \
             -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" \
             -R "select[gpu_model0==${GPU_MODEL}]" \
             -R "select[gpu_mtotal0>=30000]" \
             "source ~/.bashrc; \
             conda activate precedent; \
             python src/train.py --dataset scan --train ${train_dir}/${experiment}/tasks_train_simple.txt --dev ${train_dir}/${experiment}/tasks_train_simple.txt --test ${train_dir}/${experiment}/tasks_test_simple.txt  --model model5/${train_dir}/${arch}/${experiment} --embed_dim 100 \
             --src_hs ${size} \
             --trg_hs ${size} \
             --dropout 0.5 \
             --src_layer 2 \
             --trg_layer 2 \
             --max_norm 5 \
             --shuffle \
             --arch ${arch} \
             --gpuid 0 \
             --estop 1e-8 \
             --epochs 50 \
             --bs 64 \
             --cleanup_anyway \
             --total_eval 1 \
             --max_steps 100000"
      done
    done
  done
done

