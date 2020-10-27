import os
import os.path as osp
import sys
import yaml
import csv
from autoparrots.utils.fileio import dump


def get_frames(configs_dir):
    configs = dict()
    sum = 0
    for f in os.listdir(configs_dir):
        if f.endswith(".yaml"):
            config_path = osp.join(configs_dir, f)
            config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
            configs[f[:-5]] = config
            print("get {:20} frame {:3} models".format(f[:-5], len(config)))
            sum += len(config)
    print("get {:30} models".format(sum))
    return configs


def read_csv(r_path):
    content_list = []
    with open(r_path) as r:
        reader = csv.reader(r)
        for row in reader:
            content_list.append(row)
    return content_list


def write_csv(w_path, ops, env):
    with open(w_path, 'w', newline="") as w:
        writer = csv.writer(w)
        for op in ops:
            writer.writerow(op)


def save(data_dir, frames, env):
    if not os.path.exists(data_dir):
        save_values = []
        save_values.append(['frame', 'model', 'items', env])
        for frame, models in frames.items():
            for model, items in models.items():
                for item, v in items.items():
                    value = [frame, model, item, v]
                    save_values.append(value)
        write_csv(data_dir, save_values, env)
    else:
        read = read_csv(data_dir)
        read_map = dict()
        for idx, val in enumerate(read):
            read_map[val[0] + val[1] + val[2]] = idx
        read[0].append(env)
        for frame, models in frames.items():
            for model, items in models.items():
                for item, v in items.items():
                    key = frame + model + item
                    if key in read_map:
                        miss_num = len(read[0]) - len(read[read_map[key]]) - 1
                        for m in range(miss_num):
                            read[read_map[key]].append(None)    
                        read[read_map[key]].append(v)
                    else:
                        value = [frame, model, item]
                        for i in range(len(read[0]) - 4):
                            value.append(None)
                        value.append(v)
                        read.append(value)
        write_csv(data_dir, read, env)


def del_files(path):
    for name in os.listdir(path):
        if name.endswith(".yaml"):
            os.remove(os.path.join(path, name))
            print("delete file {}/{}".format(path, name))


if __name__ == '__main__':
    assert len(sys.argv) == 3, "INPUTERROR: python autotest/benchmark_yamls_to_csv.py pat20200821 True/False"
    title = sys.argv[1]
    this_dir = osp.dirname(os.path.abspath(__file__))
    frames_dir = osp.join(this_dir, 'benchmark')
    data_dir = osp.join(frames_dir, 'data')
    frames = get_frames(frames_dir)
    dump(frames, data_dir + '/' + title + '.yaml', file_format='yaml', default_flow_style=False)

    if sys.argv[2] == 'True':
        save(data_dir + '/benchmark.csv', frames, title)
        del_files(frames_dir)
