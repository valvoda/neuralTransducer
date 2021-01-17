"""
collect results for all models
place them in an excel
experiment/dataset/result_model1/result_model2...
"""
import argparse
import os
import csv
import re
import collections
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Display():
    def __init__(self, path):
        self.path = path

    def run(self):
        paths = self.get_paths()
        all_acc = self.get_acc(paths)

        for i, j in zip(paths,all_acc):
            print(i, j)

        self.save_sheet(paths, all_acc)

    def save_sheet(self, paths, all_acc):
        res_dic = dict()
        for p, a in zip(paths, all_acc):
            node = p.split('/')[1]
            node = int(re.sub(r'exp1_run', '', node))
            i = int(p.split('/')[-1].strip('.log'))
            if node in res_dic.keys():
                res_dic[node].append(a)
            else:
                res_dic[node] = [a]

        res_dic = collections.OrderedDict(sorted(res_dic.items()))

        with open('results.csv', mode='w') as r_file:
            r_writer = csv.writer(r_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for key in res_dic.keys():
                nodes = [key]
                nodes += res_dic[key]
                r_writer.writerow(nodes)

        # for key in res_dic:
        #     for i in res_dic[key]:
        #         if i != -1:
        #             plt.scatter(key, i, label=key)

        # import pdb
        # pdb.set_trace()

        df = pd.DataFrame.from_dict(res_dic, orient='index')
        df.index.rename('Observation', inplace=True)

        stacked = df.stack().reset_index()
        stacked.rename(columns={'level_1': 'Person', 0: 'Value'}, inplace=True)

        g = sns.regplot(data=stacked, x='Observation', y='Value', scatter=True, order=2)
        g.legend_.remove()

        plt.savefig('all_results.png')
        # plt.close()
        #
        # for key in res_dic:
        #     for i in res_dic[key]:
        #         if i != -1 and key <= 100:
        #             plt.scatter(key, i, label=key)
        #
        # plt.savefig('10_results.png')
        # plt.close()

    def get_acc(self, paths):
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

    def get_paths(self):
        all_paths = []
        src_path = self.path
        dirs = os.listdir(src_path)
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
    args = parser.parse_args()
    ds = Display(args.path)
    ds.run()