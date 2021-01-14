"""
collect results for all models
place them in an excel
experiment/dataset/result_model1/result_model2...
"""

import os

def run():
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
                        print(exp_path)
                    # try:
                    #     with open(exp_path, 'r') as f:
                    #         lines = f.readlines()
                    #         lines = [l.strip("\n") for l in lines]
                    #         print(lines[-1])
                    # except:
                    #     pass
            except:
                exp_path = os.path.join(n_path, nd)
                if exp_path.endswith(".log"):
                    print(exp_path)

if __name__ == '__main__':
    run()