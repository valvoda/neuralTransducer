#!/bin/bash

# Submit job
for arch in soft
#hard softinputfeed largesoftinputfeed approxihard \
#approxihardinputfeed hmm hmmfull transformer universaltransformer \
#tagtransformer taguniversaltransformer
do
  for experiment in {0..100}
  do
    for size in 300
    do
      # 10exp1_run 20exp1_run 30exp1_run 40exp1_run 60exp1_run 70exp1_run 80exp1_run 90exp1_run 50exp1_run 100exp1_run 200exp1_run 400exp1_run 600exp1_run 800exp1_run 1000exp1_run 200exp1_run 300exp1_run 500exp1_run 700exp1_run 900exp1_run
      # 50exp2_run 100exp2_run 200exp2_run 400exp2_run 600exp2_run 800exp2_run 1000exp2_run 200exp2_run 300exp2_run 500exp2_run 700exp2_run 900exp2_run
      for train_dir in 21exp1_run 22exp1_run 23exp1_run 24exp1_run 25exp1_run 26exp1_run 27exp1_run 28exp1_run 29exp1_run 30exp1_run 31exp1_run 32exp1_run 33exp1_run 34exp1_run 35exp1_run 36exp1_run 37exp1_run 38exp1_run 39exp1_run
      do
        # Submit job
        TRAIN_DIR=$train_dir EXPERIMENT=$experiment SIZE=$size ARCH=$arch sbatch run.wilkes3
      done
    done
  done
done