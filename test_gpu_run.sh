#!/bin/bash

python src/train.py --dataset scan --train Dataset/29exp1_run/0/tasks_train_simple.txt --dev Dataset/29exp1_run/0/tasks_train_simple.txt --test Dataset/29exp1_run/0/tasks_test_simple.txt --src_hs 300 --trg_hs 300 --dropout 0.5 --src_layer 2 --trg_layer 2 --max_norm 5 --shuffle --arch soft --gpuid 0 --estop 1e-8 --epochs 5 --bs 512 --cleanup_anyway --total_eval 1 --max_steps 100 --model model/300/29