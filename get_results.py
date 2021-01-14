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
            nn_path = os.path.join(n_path, nd)
            for nnd in nn_path:
                exp_path = os.path.join(nn_path, nnd)
                print(exp_path)
                # try:
                #     with open(exp_path, 'r') as f:
                #         lines = f.readlines()
                #         lines = [l.strip("\n") for l in lines]
                #         print(lines[-1])
                # except:
                #     pass
if __name__ == '__main__':
    run()