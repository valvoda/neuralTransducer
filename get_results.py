"""
collect results for all models
place them in an excel
experiment/dataset/result_model1/result_model2...
"""

import os

def run():
    paths = get_paths()
    all_acc = get_acc(paths)
    for i, j in zip(paths,all_acc):
        print(i, j)

def get_acc(paths):
    all_acc = []
    for p in paths:
        try:
            with open(p, 'r') as f:
                lines = f.readlines()
                lines = [l.strip("\n") for l in lines]
                result = lines[-1]
                acc = result.split(" acc ")[1].split(" ")[0]
                all_acc.append(acc)
        except:
            all_acc.append(-1)

    return all_acc


def get_paths():
    all_paths = []
    src_path = "./model"
    dirs = os.listdir("./model")
    for d in dirs:
        n_path = os.path.join(src_path, d)
        n_dirs = os.listdir(n_path)
        for nd in n_dirs:
            try:
                nn_path = os.path.join(n_path, nd)
                nn_dirs = os.listdir(nn_path)
                for nnd in nn_dirs:
                    exp_path = os.path.join(nn_path, nnd)
                    if exp_path.endswith(".log"):
                        all_paths.append(exp_path)
            except:
                exp_path = os.path.join(n_path, nd)
                if exp_path.endswith(".log"):
                    all_paths.append(exp_path)

    return all_paths

if __name__ == '__main__':
    run()