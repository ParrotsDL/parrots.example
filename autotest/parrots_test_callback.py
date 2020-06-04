import os
import os.path as osp
import sys
import re
import time
import datetime

import yaml
import numpy as np


callback_funcs = dict()

if os.environ.get('LOG_STREAM_DEBUG') is not None:
    log_stream = open(os.environ.get('LOG_STREAM_DEBUG'), 'r').readlines()
else:
    log_stream = sys.stdin


def register_callfunc(func):
    callback_funcs[func.__name__] = func

    def warp_func(*args, **kwargs):
        return func(*args, **kwargs)

    return warp_func


@register_callfunc
def pod_func(done_flag="copypaste:|Thanks",
             iter_speed_flag="Progress:\[100\.0\%\](.*)batch_time:[0-9]*.[0-9]*\(([0-9]*.[0-9]*)\)",
             acc_flag1="\"bbox\.AP\": ([0-9]*.[0-9]*)|mAP: ([0-9]*.[0-9]*)",
             acc_flag2="\"segm\.AP\": ([0-9]*.[0-9]*)|\"keypoints\.AP\": ([0-9]*.[0-9]*)",
             ips_flag="IP:(.+)",
             **args):
    """ log_file: read from stdin as default
        ret(dict): results of analysis metrics
    """
    ret = {}
    ret.update(**args)
    ret['is_done'] = False
    ret['iter_speed'] = 'none'
    ret['bbox_ap'] = 'none'
    ret['seg_ap'] = 'none'
    ret['ips'] = 'none'
    for line in log_stream:
        if ret['is_done'] is False:
            is_done = re.search(done_flag, line)
            if is_done is not None:
                ret['is_done'] = True
        if ret['iter_speed'] == 'none':
            iter_speed = re.search(iter_speed_flag, line)
            if iter_speed is not None:
                ret['iter_speed'] = iter_speed.group(2)
        acc1 = re.search(acc_flag1, line)
        if acc1 is not None:
            ret['bbox_ap'] = acc1.group()
        acc2 = re.search(acc_flag2, line)
        if acc2 is not None:
            ret['seg_ap'] = acc2.group()
        if ret['ips'] == 'none':
            ips = re.search(ips_flag, line)
            if ips is not None:
                ret['ips'] = ips.group(1)
    return ret


@register_callfunc
def alphatrion_func(done_flag="Pipeline is Done",
                    iter_speed_flag="epoch \[1\/\d+\], sub_epoch \[1\/1\], step \[200(.*)batch \[([0-9]+\.[0-9]{3})",
                    acc_flag="valid_top1 \[[0-9]+\.[0-9]{4} \(([0-9]+\.[0-9]{4})",
                    start_flag="[0-9]{4}-[0-2][0-9]-[0-3][0-9]-[0-2][0-9]:[0-6][0-9]:[0-6][0-9]",
                    ips_flag="nodelist \[(.+)\]",
                    **args):
    ret = {}
    ret.update(**args)
    ret['is_done'] = False
    ret['iter_speed'] = 'none'
    ret['valid_top1'] = 'none'
    ret['start'] = 'none'
    ret['ips'] = 'none'
    for line in log_stream:
        if ret['is_done'] is False:
            is_done = re.search(done_flag, line)
            if is_done is not None:
                ret['is_done'] = True
        if ret['iter_speed'] == 'none':
            iter_speed = re.search(iter_speed_flag, line)
            if iter_speed is not None:
                ret['iter_speed'] = iter_speed.group(2)
        acc = re.search(acc_flag, line)
        if acc is not None:
            ret['valid_top1'] = acc.group(1)
        if ret['start'] == 'none':
            start = re.search(start_flag, line)
            if start is not None:
                ret['start'] = start.group()
        if ret['ips'] == 'none':
            ips = re.search(ips_flag, line)
            if ips is not None:
                ret['ips'] = ips.group(1)
    return ret


@register_callfunc
def seg_nas_func(done_flag="Pipeline is Done",
                 acc_flag="main\.py\#(.*)\((.*)\, ([0-9]*.[0-9]*)\)",
                 start_flag="[0-9]{4}-[0-2][0-9]-[0-3][0-9] [0-2][0-9]:[0-6][0-9]:[0-6][0-9]",
                 end_flag="[0-9]{4}-[0-2][0-9]-[0-3][0-9] [0-2][0-9]:[0-6][0-9]:[0-6][0-9]",
                 ips_flag="IP:(.+)",
                 **args):
    ret = {}
    ret.update(**args)
    ret['is_done'] = False
    ret['acc'] = 'none'
    ret['start'] = 'none'
    ret['end'] = 'none'
    ret['ips'] = 'none'
    ret['total_time'] = 'none'
    for line in log_stream:
        if ret['is_done'] is False:
            is_done = re.search(done_flag, line)
            if is_done is not None:
                ret['is_done'] = True
        acc = re.search(acc_flag, line)
        if acc is not None:
            ret['acc'] = acc.group(3)
        if ret['start'] == 'none':
            start = re.search(start_flag, line)
            if start is not None:
                ret['start'] = start.group()
        end = re.search(end_flag, line)
        if end is not None:
            ret['end'] = end.group()
        if ret['ips'] == 'none':
            ips = re.search(ips_flag, line)
            if ips is not None:
                ret['ips'] = ips.group(1)
    if ret['start'] != 'none' and ret['end'] != 'none':
        date1 = time.strptime(ret['start'], "%Y-%m-%d %H:%M:%S")
        date2 = time.strptime(ret['end'], "%Y-%m-%d %H:%M:%S")
        date1 = datetime.datetime(
            date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
        date2 = datetime.datetime(
            date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
        total_time = date2-date1
        total_time = total_time.days*24+total_time.seconds/3600
        ret['total_time'] = total_time
    return ret


@register_callfunc
def ssd_func(done_flag="Pipeline is Done",
             iter_speed_flag="epoch \[1\/\d+\], step \[1000(.*)batch \[([0-9]+\.[0-9]{3})",
             acc_flag="valid_mAP \[[0-9]+\.[0-9]{4} \(([0-9]+\.[0-9]{4})",
             start_flag="[0-9]{4}-[0-2][0-9]-[0-3][0-9]-[0-2][0-9]:[0-6][0-9]:[0-6][0-9]",
             end_flag="[0-9]{4}-[0-2][0-9]-[0-3][0-9]-[0-2][0-9]:[0-6][0-9]:[0-6][0-9]",
             ips_flag="hostname is:(.+)",
             **args):
    ret = {}
    ret.update(**args)
    ret['is_done'] = False
    ret['iter_speed'] = 'none'
    ret['valid_mAP'] = 'none'
    ret['start'] = 'none'
    ret['end'] = 'none'
    ret['ips'] = 'none'
    ret['total_time'] = 'none'
    for line in log_stream:
        if ret['is_done'] is False:
            is_done = re.search(done_flag, line)
            if is_done is not None:
                ret['is_done'] = True
        if ret['iter_speed'] == 'none':
            iter_speed = re.search(iter_speed_flag, line)
            if iter_speed is not None:
                ret['iter_speed'] = iter_speed.group(2)
        acc = re.search(acc_flag, line)
        if acc is not None:
            ret['valid_mAP'] = acc.group(1)
        if ret['start'] == 'none':
            start = re.search(start_flag, line)
            if start is not None:
                ret['start'] = start.group()
        end = re.search(end_flag, line)
        if end is not None:
            ret['end'] = end.group()
        if ret['ips'] == 'none':
            ips = re.search(ips_flag, line)
            if ips is not None:
                ret['ips'] = ips.group(1)
    if ret['start'] != 'none' and ret['end'] != 'none':
        date1 = time.strptime(ret['start'], "%Y-%m-%d %H:%M:%S")
        date2 = time.strptime(ret['end'], "%Y-%m-%d %H:%M:%S")
        date1 = datetime.datetime(
            date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
        date2 = datetime.datetime(
            date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
        total_time = date2-date1
        total_time = total_time.days*24+total_time.seconds/3600
        ret['total_time'] = total_time
    return ret


@register_callfunc
def mild_func(done_flag="Pipeline is Done",
             f_iter_speed_flag="F\-Time\ \d{1,5}\.\d{2}\ \(\d{1,5}\.\d{2}\)",
             b_iter_speed_flag="B\-Time\ \d{1,5}\.\d{2}\ \(\d{1,5}\.\d{2}\)",
             d_iter_speed_flag="Data\-Time\ \d{1,5}\.\d{2}\ \(\d{1,5}\.\d{2}\)",
             mimic_loss_flag='Mimic\ Loss\ \d{1,5}\.\d{2}\ \(\d{1,5}\.\d{2}\)',
             cls_loss_flag='Cls\ Loss\ \d{1,5}\.\d{2}\ \(\d{1,5}\.\d{2}\)',
             ips_flag="hostname is:(.+)",
             **args):
    ret = {}
    ret.update(**args)
    keys = ['is_done', 'fwd_iter_speed', 'bwd_iter_speed', 'data_iter_speed',
            'mimic_loss', 'cls_loss', 'ips']
    flags = [done_flag, f_iter_speed_flag, b_iter_speed_flag, d_iter_speed_flag,
             mimic_loss_flag, cls_loss_flag, ips_flag]
    for key in keys:
        ret[key] = None
    for line in sys.stdin:
        for key, flag in zip(keys, flags):
            rv = re.search(flag, line)
            if rv is None:
                # not found flag in this line
                continue
            ret[key] = rv.group(0)
    return ret

@register_callfunc
def alphatrion_nas_func(done_flag="Pipeline is Done",
                 max_sce_flag="max_origin_valid_sce (\[.*\])",
                 max_top1_flag="max_origin_valid_top1 (\[.*\])",
                 max_top5_flag="max_origin_valid_top5 (\[.*\])",
                 ref_sce_flag="ref_origin_valid_sce (\[.*\])",
                 ref_top1_flag="ref_origin_valid_top1 (\[.*\])",
                 ref_top5_flag="ref_origin_valid_top5 (\[.*\])",
                 min_sce_flag="min_origin_valid_sce (\[.*\])",
                 min_top1_flag="min_origin_valid_top1 (\[.*\])",
                 min_top5_flag="min_origin_valid_top5 (\[.*\])",
                 iter_speed_flag="lr \[.*\], batch \[(.*)\], compute",
                 start_flag="[0-9]{4}-[0-2][0-9]-[0-3][0-9]-[0-2][0-9]:[0-6][0-9]:[0-6][0-9]",
                 end_flag="[0-9]{4}-[0-2][0-9]-[0-3][0-9]-[0-2][0-9]:[0-6][0-9]:[0-6][0-9]",
                 ips_flag="nodelist \[(.+)\]",
                 **args):
    ret = {}
    ret.update(**args)
    ret['is_done'] = False

    ret['max_origin_valid_sce'] = 'none'
    ret['max_origin_valid_top1'] = 'none'
    ret['max_origin_valid_top5'] = 'none'
    ret['ref_origin_valid_sce'] = 'none'
    ret['ref_origin_valid_top1'] = 'none'
    ret['ref_origin_valid_top5'] = 'none'
    ret['min_origin_valid_sce'] = 'none'
    ret['min_origin_valid_top1'] = 'none'
    ret['min_origin_valid_top5'] = 'none'

    ret['iter_speed'] = 'none'
    ret['start'] = 'none'
    ret['end'] = 'none'
    ret['ips'] = 'none'
    ret['total_time'] = 'none'

    iter_speeds = []
    for line in log_stream:
        if ret['is_done'] is False:
            is_done = re.search(done_flag, line)
            if is_done is not None:
                ret['is_done'] = True
        if ret['start'] == 'none':
            start = re.search(start_flag, line)
            if start is not None:
                ret['start'] = start.group()
        end = re.search(end_flag, line)
        if end is not None:
            ret['end'] = end.group()
        if ret['ips'] == 'none':
            ips = re.search(ips_flag, line)
            if ips is not None:
                ret['ips'] = ips.group(1)
        
        iter_speed_f = re.search(iter_speed_flag, line)
        if iter_speed_f is not None:
            iter_speeds.append(iter_speed_f.group(1))

        if ret['max_origin_valid_sce'] == 'none':
            max_sce = re.search(max_sce_flag, line)
            if max_sce is not None:
                ret['max_origin_valid_sce'] = max_sce.group()
        
        if ret['max_origin_valid_top1'] == 'none':
            max_top1 = re.search(max_top1_flag, line)
            if max_top1 is not None:
                ret['max_origin_valid_top1'] = max_top1.group()
        
        if ret['max_origin_valid_top5'] == 'none':
            max_top5 = re.search(max_top5_flag, line)
            if max_top5 is not None:
                ret['max_origin_valid_top5'] = max_top5.group()

        if ret['ref_origin_valid_sce'] == 'none':
            ref_sce = re.search(ref_sce_flag, line)
            if ref_sce is not None:
                ret['ref_origin_valid_sce'] = ref_sce.group()
        
        if ret['ref_origin_valid_top1'] == 'none':
            ref_top1 = re.search(ref_top1_flag, line)
            if ref_top1 is not None:
                ret['ref_origin_valid_top1'] = ref_top1.group()
        
        if ret['ref_origin_valid_top5'] == 'none':
            ref_top5 = re.search(ref_top5_flag, line)
            if ref_top5 is not None:
                ret['ref_origin_valid_top5'] = ref_top5.group()

        if ret['min_origin_valid_sce'] == 'none':
            min_sce = re.search(min_sce_flag, line)
            if min_sce is not None:
                ret['min_origin_valid_sce'] = min_sce.group()
        
        if ret['min_origin_valid_top1'] == 'none':
            min_top1 = re.search(min_top1_flag, line)
            if min_top1 is not None:
                ret['min_origin_valid_top1'] = min_top1.group()
        
        if ret['min_origin_valid_top5'] == 'none':
            min_top5 = re.search(min_top5_flag, line)
            if min_top5 is not None:
                ret['min_origin_valid_top5'] = min_top5.group()

    if ret['start'] != 'none' and ret['end'] != 'none':
        date1 = time.strptime(ret['start'], "%Y-%m-%d-%H:%M:%S")
        date2 = time.strptime(ret['end'], "%Y-%m-%d-%H:%M:%S")
        date1 = datetime.datetime(
            date1[0], date1[1], date1[2], date1[3], date1[4], date1[5])
        date2 = datetime.datetime(
            date2[0], date2[1], date2[2], date2[3], date2[4], date2[5])
        total_time = date2-date1
        total_time = total_time.days*24+total_time.seconds/3600
        ret['total_time'] = total_time
    iter_speeds = [float(t) for t in iter_speeds]
    ret['iter_speed'] = np.mean(iter_speeds)
    return ret


@register_callfunc
def nas_lite_func(done_flag="Pipeline is Done",
                  iter_speed_flag="INFO] Iter: \[100\/.*]	Time [0-9]*.[0-9]* \(([0-9]*.[0-9]*)\)	Data",
                  prec_flag="Prec@1 [0-9]*.[0-9]* \(([0-9]*.[0-9]*)\)	Prec@5 [0-9]*.[0-9]* \(([0-9]*.[0-9]*)\)",
                  ips_flag="node_list: (.+)",
                  **args):
    """ log_file: read from stdin as default
        iter_speed_flag: flag for 100 iter time
        prec_flag: flag for Prec@1 and Prec@5
        ret(dict): results of analysis metrics
    """
    ret = {}
    ret.update(**args)
    ret['is_done'] = False
    ret['iter_speed'] = 'none'
    ret['prec1'] = 'none'
    ret['prec5'] = 'none'
    ret['ips'] = 'none'
    for line in log_stream:
        # print(line)
        if ret['is_done'] is False:
            is_done = re.search(done_flag, line)
            if is_done is not None:
                ret['is_done'] = True
        if ret['iter_speed'] == 'none':
            iter_speed = re.search(iter_speed_flag, line)
            if iter_speed is not None:
                ret['iter_speed'] = iter_speed.group(1)
        prec = re.search(prec_flag, line)
        if prec is not None:
            ret['prec1'] = prec.group(1)
            ret['prec5'] = prec.group(2)
        if ret['ips'] == 'none':
            ips = re.search(ips_flag, line)
            if ips is not None:
                ret['ips'] = ips.group(1)
    return ret


def callback_wapper(func_name, **args):
    ret_dict = callback_funcs[func_name](**args)
    # Required: encoding to yaml format and output to stdout,
    #           so that it can be auto-parsed to summary.
    print(yaml.dump(ret_dict))


def collect_config(framework, model_name):
    this_dir = osp.dirname(os.path.abspath(__file__))
    configs_dir = osp.join(this_dir, 'configs')
    configs = dict()
    for f in os.listdir(configs_dir):
        config_path = osp.join(configs_dir, f)
        configs.update(
            **yaml.load(open(config_path, 'r'), Loader=yaml.Loader))
    key = framework + '_' + model_name
    return configs[key]


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        config = collect_config(sys.argv[1], sys.argv[2])
        callback_wapper(config['func'], **config['args'],
                        thresh=config['thresh'])
