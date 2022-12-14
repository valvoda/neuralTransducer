# Learning Transductions to Test Systematic Compositionality

This repo builds on a neural transducer work of Shijie Wu: https://github.com/shijie-wu/neural-transducer

The aditional code supports the 'Learning Transductions to Test Systematic Compositionality' paper:
https://arxiv.org/pdf/2208.08195.pdf

## Get Started Quickly

#1 Create and activate environment (conda) with: `environment.yml`
- `conda env create --name neuralTrans --file=environment.yml`
- `conda activate neuralTrans`

#2 Generate datasets by running:
- `python dataset_generator.py`

#3 Benchmark a dataset by running:
- `python src/train.py --dataset scan --train 20000/10/0/tasks_train_simple.txt --dev 20000/10/0/tasks_train_simple.txt --test 20000/10/0/tasks_test_simple.txt --model LSTM/10/0 --arch soft`
- Look at example folder and src/train.py for all the hyper-parameters you can set.

## License

MIT
