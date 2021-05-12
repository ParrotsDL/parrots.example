import matplotlib.pyplot as plt
import os
import re
import math
import time
import numpy as np
import argparse


def plot_line(iter_list, loss_list, acc1_list, acc5_list, file_path, title="Net"):
    # fig, ax1 = plt.subplots()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    
    p1 = ax1.plot(iter_list, loss_list, 'royalblue', label="loss")
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    # ax1.set_ylim(0, 10)
    

    ax12 = ax1.twinx()
    p12 = ax12.plot(iter_list, acc1_list, 'darkorange', label="acc1")
    # ax12.plot(iter_list, acc5_list, 'o', label="acc5")
    ax12.set_ylabel('acc(%)')
    ax12.set_ylim(0, 100)
    # ax1.legend(loc=0)
    # ax12.legend(loc=1)

    plt.legend(p1 + p12, ["loss", "acc1"])

    cur_epoch = max(iter_list) // 10010
    plt.xticks([(i + 1) * 10010 for i in range(cur_epoch)], [i + 1 for i in range(cur_epoch)])
    
    fig.savefig(file_path + ".png", dpi=600)
    # fig.savefig(file_path + ".svg")

    
def read_log(log_path):
    iter_list = []
    loss_list = []
    acc1_list = []
    acc5_list = []
    with open(log_path) as file:
        for f in file:
            line = f.strip()
            walk = re.findall('\[(.*?)\]', line)
            if len(walk) < 2: continue
            cur_epoch = int(walk[0].split("/")[0])
            cur_iter = int(walk[1].split("/")[0])
            total_iter = int(walk[1].split("/")[1])
            
            data = re.findall('\((.*?)\)', line)
            if len(data) <= 0: continue
            loss = float(data[2])
            acc1 = float(data[3])
            acc5 = float(data[4])
            # print(data)
            # print(f"{cur_epoch}, {cur_iter}/{total_iter}, {loss}, {acc1}, {acc5}")
            iter_list.append((cur_epoch - 1) * total_iter + cur_iter)
            # iter_list.append((cur_epoch)
            loss_list.append(loss)
            acc1_list.append(acc1)
            acc5_list.append(acc5)
    return (iter_list, loss_list, acc1_list, acc5_list)

def sleep_time(hour, min, sec):
    return hour * 3600 + min * 60 + sec

def main(args):
    second = sleep_time(0, 1, 0)
    while True:

        print(time.localtime(time.time()))
        
        (iter_list, loss_list, acc1_list, acc5_list) = read_log(args.log_path)
        
        plot_line(iter_list, loss_list, acc1_list, acc5_list, args.log_path, args.title)

        time.sleep(second)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, help='path to log file')
    parser.add_argument('--title', type=str, default='Net', help='fig title')
    args = parser.parse_args()
    main(args)