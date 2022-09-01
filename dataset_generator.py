import networkx as nx
import numpy as np
from pynini import Fst, Arc
import random
from tqdm import tqdm
import pickle
from collections import Counter
from pathlib import Path


def generate_FST(node_size):
    while True:
        n = node_size
        adjacency = np.random.randint(0, 2, (n, n))
        G = nx.DiGraph(adjacency)

        labels = dict()
        for i in range(len(adjacency)):
            labels[i] = str(i)

        strong_index = []
        for i in nx.strongly_connected_components(G):
            #     print(i)
            if list(i)[-1] != 0:
                gen = nx.all_simple_paths(G, source=0, target=list(i)[-1])
                # if sum(1 for _ in gen) > 0:
                try:
                    next(gen)
                    strong_index.append(1)
                except:
                    strong_index.append(0)

        fst = True
        if 0 in strong_index:
            fst = False

        if fst:
            generate = False
            print(adjacency)
            #  nx.draw(G, labels=labels)

            branches = [i[0] for i in G.edges]
            minimum_input_alphabet = max(list(Counter(branches).values()))
            # if minimum_input_alphabet < 10:
            #     minimum_input_alphabet = 10

            input_alphabet_size = np.random.randint(minimum_input_alphabet, len(G.edges))
            # output_alphabet_size = np.random.randint(1, len(G.edges))
            # input_alphabet_size = 30
            output_alphabet_size = 30 # was 10

            output_alphabet = random.sample(range(output_alphabet_size, output_alphabet_size*2), output_alphabet_size)

            input_alphabet = random.sample(range(0, input_alphabet_size), input_alphabet_size)
            input_alphabet_cpy = input_alphabet.copy()

            # empty_emmission = random.sample([0,1], 1)
            empty_emmission = np.random.randint(2, size=1)
            if empty_emmission == 1:
                print("Empty Emissions: YES")
                output_alphabet.pop(0)
                output_alphabet.append(-1)
            else:
                print("Empty Emissions: NO")

            print("Graph_nodes:", len(adjacency), "edges:", len(G.edges))
            print("input_alpha", input_alphabet, len(input_alphabet))
            print("output_alpha", output_alphabet, len(output_alphabet))

            fst = Fst(arc_type='standard')
            fst.add_state()
            fst.set_start(0)
            cnt = 0

            FST_dic = dict()
            node_dic = dict()
            for i in G.edges:
                FST_dic[cnt] = node_dic
                while i[0] > cnt:
                    fst.add_state()
                    cnt += 1
                    input_alphabet_cpy = input_alphabet.copy()
                    node_dic = dict()

                in_index = random.randint(0, len(input_alphabet_cpy) - 1)
                in_label = input_alphabet_cpy.pop(in_index)

                out_index = random.randint(0, len(output_alphabet) - 1)
                out_label = output_alphabet[out_index]

                node_dic[i[1]] = (in_label, out_label)

                fst.add_arc(i[0], Arc(in_label, out_label, 0, i[1]))

            fst.minimize(allow_nondet=False)

            return fst, FST_dic, input_alphabet, output_alphabet, node_size, len(G.edges), adjacency


def generate_dataset(FST_dic, max_len):
    #  print(FST_dic)
    inputs = []
    outputs = []
    node_cvr = [0]
    for _ in tqdm(range(100000)):
        max_size = random.randint(1, max_len)
        input_path = []
        output_path = []
        run = True
        i = 0
        cnt = 0
        stopper = False
        while run and cnt < max_size and stopper == False:
            try:
                index = random.randint(0, (len(FST_dic[i])) - 1)
                path = list(FST_dic[i].keys())[index]
                input_path.append(FST_dic[i][path][0])
                if FST_dic[i][path][1] != -1:
                    output_path.append(FST_dic[i][path][1])
                i = path
                node_cvr.append(i)
                cnt += 1
                if random.randint(0, 10) == 1:
                    stopper = True
            except:
                run = False
        inputs.append(input_path)
        outputs.append(output_path)

    test = [str(i) for i in inputs]
    samples = list(set(test))
    print("size of exp:", len(samples))

    dic_flat = {}
    for i, j in zip(inputs, outputs):
        dic_flat[str(i)] = j

    import ast
    inputs = [ast.literal_eval(i) for i in list(dic_flat.keys())]
    outputs = list(dic_flat.values())

    print("size of exp:", len(inputs))
    print("node_covered:", len(list(set(node_cvr))))

    for i, j in zip(inputs[:10], outputs[:10]):
        print(i[:5], "transduces:", j[:5])
    return inputs, outputs


def find_split(outputs):
    for len_n in range(5,45):
        small_lengths = [i for i in outputs if len(i) <= len_n]
        long_lengths = [i for i in outputs if len(i) > len_n]
        split = len(long_lengths)/ (len(long_lengths) + len(small_lengths))
        if split>0.15 and split<0.25:
            print(split)
            return len_n


def save_data(name, train_inputs, train_outputs, test_inputs, test_outputs,
              fst, FST_dic, input_alphabet, output_alphabet, nodes, edges, out_path):
    #     make directory with number, save train file, test file, info file and fst + fst_dic

    out_path = out_path + "/" + name
    print(out_path)
    Path(out_path).mkdir(parents=True, exist_ok=True)
    train_file = "tasks_train_simple.txt"
    test_file = "tasks_test_simple.txt"
    with open(out_path + "/" + train_file, 'w') as f:
        for i, o in zip(train_inputs, train_outputs):
            in_command = stringify(i)
            out_command = stringify(o)
            f.write("IN: " + in_command + "OUT: " + out_command + "\n")

    with open(out_path + "/" + test_file, 'w') as f:
        for i, o in zip(test_inputs, test_outputs):
            in_command = stringify(i)
            out_command = stringify(o)
            f.write("IN: " + in_command + "OUT: " + out_command + "\n")

    with open(out_path + "/info.pickle", 'wb') as f:
        a = [fst, FST_dic, input_alphabet, output_alphabet, nodes, edges]
        pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)


def split_data(split, inputs, outputs):
    train_inputs = []
    test_inputs = []
    train_outputs = []
    test_outputs = []
    for i, j in zip(inputs, outputs):
        if len(j) <= split:
            train_inputs.append(i)
            train_outputs.append(j)
        else:
            test_inputs.append(i)
            test_outputs.append(j)

    return train_inputs, test_inputs, train_outputs, test_outputs


def stringify(input_list):
    text = ""
    for i in input_list:
        text += str(i) + " "
    return text


def run(dataset_n=100, node_n=50, seq_len_max=50, dataset_name="test", dataset_size=40000):
    cnt = 0
    adjacency_dic = dict()

    out_path = './' + str(dataset_size) + '/' + dataset_name
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # Generate 100 SCAN like datasets
    while cnt < dataset_n:
        node_size = node_n
        max_length = seq_len_max

        fst, FST_dic, input_alphabet, output_alphabet, nodes, edges, adjacency = generate_FST(node_size)

        in_al = []
        out_al = []
        for v in FST_dic.values():
            for i in v.values():
                in_al.append(i[0])
                out_al.append(i[1])

        in_len = len(set(in_al))
        out_len = len(set(out_al))

        # Check if the same graph dataset hasn't already been created.
        try:
            adjacency_dic[str(adjacency)]
        except:
            adjacency_dic[str(adjacency)] = 1

            inputs, outputs = generate_dataset(FST_dic, max_length)
            assert len(inputs) == len(outputs)

            # Check if the lenght of dataset matches SCAN
            if len(inputs) >= dataset_size:
                print("Dataset")
                inputs = inputs[:dataset_size]
                outputs = outputs[:dataset_size]
                #     Experiment 1: 80/20 split
                train_size = int(len(inputs) / 100 * 80)
                train_inputs = inputs[:train_size]
                test_inputs = inputs[train_size:]
                train_outputs = outputs[:train_size]
                test_outputs = outputs[train_size:]
                save_data(str(cnt), train_inputs, train_outputs, test_inputs, test_outputs,
                          fst, FST_dic, input_alphabet, output_alphabet, nodes, edges, out_path)
                cnt += 1

def get_data(file):

    sources, targets = [], []

    with open(file, "r", encoding="utf-8") as fp:
        for line in fp.readlines():
            target_raw = line.split("OUT:")[1].strip()
            source_raw = line.split("OUT:")[0].split("IN:")[1].strip()

            target = target_raw.split(" ")
            source = source_raw.split(" ")

            sources.append(source)
            targets.append(target)

    return sources, targets

def gen_repeats(file_path, dest_path):
    source, target = get_data(file_path)
    n_source, n_target = [], []

    for j in list(range(len(source))):
        switch = np.random.randint(10, size=1)[0]
        if switch == 1 or switch == 2:
            x = source[j]
            x.append('-10')
            n_source.append(x)
            y = target[j] + target[j]
            n_target.append(y)
        elif switch == 3 or switch == 4:
            x = source[j]
            x.append('-20')
            n_source.append(x)
            y = target[j] + target[j] + target[j] + target[j]
            n_target.append(y)
        else:
            n_source.append(source[j])
            n_target.append(target[j])

    with open(dest_path, 'w') as out_f:
        for src, trg in zip(n_source, n_target):
            in_command = stringify(src)
            out_command = stringify(trg)
            out_f.write("IN: " + in_command + "OUT: " + out_command + "\n")

def make_repeats(data_path):
    f1 = "tasks_test_simple.txt"
    f2 = "tasks_train_simple.txt"
    out1 = "tasks_test_loop.txt"
    out2 = "tasks_train_loop.txt"

    for i in range(100):
        pth = data_path + "/" + str(i)
        pth1 = pth + "/" + f1
        pth2 = pth + "/" + f2
        gen_repeats(pth1, pth + "/" +out1)
        gen_repeats(pth2, pth + "/" +out2)


if __name__ == '__main__':

    for i in range(10,110,10):
        run(dataset_n=100, node_n=i, seq_len_max=50, dataset_name=str(i), dataset_size=20000)

    for i in range(21,40):
        run(dataset_n=100, node_n=i, seq_len_max=50, dataset_name=str(i), dataset_size=40000)

    data_path = "20000/20"
    make_repeats(data_path)