"""
collect results for all models
place them in an excel
experiment/dataset/result_model1/result_model2...
"""
import argparse
import os
import csv


def run():
    paths = get_paths()
    all_acc = get_acc(paths)

    for i, j in zip(paths,all_acc):
        print(i, j)

def save_sheet(paths, all_acc):

    for p in paths:
        p = p[1:]
        p = p.strip(".log")
        model_type = p.split("/")[1:]
        if len(model_type) == 2:
            data_name = "scan"
            model_name = model_type[1]
            exp_type = "exp1"
        else:
            data_name = model_type[2]
            model_name = model_type[1]
            exp_type = model_type[0]

    # with open('results.csv', mode='w') as r_file:
    #     r_writer = csv.writer(r_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     r_writer.writerow([data_name, exp_type, soft, hard, softinputfeed, largesoftinputfeed, approxihard,
    #                        approxihardinputfeed, hmm, hmmfull, transformer, universaltransformer, tagtransformer,
    #                        taguniversaltransformer])

def get_acc(paths):
    all_acc = []
    for p in paths:
        try:
            with open(p, 'r') as f:
                lines = f.readlines()
                lines = [l.strip("\n") for l in lines]
                result = lines[-1]
                acc = float(result.split(" acc ")[1].split(" ")[0])
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="model", required=False)
    run()