"""
监控子进程
"""
import os
import os.path as osp
import sys
import time
import psutil

from . import (callback_utils, callback_common, logger, kill_task, load_taskinfo)

def read_log_last(path, last_line_num=5):
    if not osp.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            off = -50
            while True:
                f.seek(off, 2)
                lines = f.readlines()
                if len(lines) >= last_line_num:
                    return lines
                off *= 2
    except Exception:
        return None
    return None


def get_hash(lines):
    if lines is None:
        return None
    lines_str = ' '.join(lines)
    return hash(lines_str)


def watch_for_kill_time_limited(framework, model, config, time_limited_flag='[E] Time limit exceeded'):
    time.sleep(60)
    # wait for job_pid, job_log_path
    job_pid = None
    slurm_job_id = None
    job_log_path = None
    workdir = None
    name = None
    job_wait_to_run_time_thresh = 1  # wait one hour
    start_time = time.time()
    while True:
        time.sleep(30)
        interval_time = time.time() - start_time
        if ((not job_pid or
             not job_log_path or
             not workdir or
             not name or
             not slurm_job_id) and
                interval_time >= job_wait_to_run_time_thresh * 60 * 60):
            break

        # get job pid
        try:
            job_pid = int(os.environ['pid'])
        except Exception:
            job_pid = None
        # get job wait_to_run_time_thresh
        try:
            job_wait_to_run_time_thresh = int(
                os.environ['wait_to_run_time_thresh'])
        except Exception:
            job_wait_to_run_time_thresh = 1
        # get job log path
        job_log_path = os.environ['log_path']
        # get job workdir
        try:
            workdir = os.environ['workdir']
            workdir = os.sep.join(workdir.split(os.sep)[:-2])
        except:
            workdir = None
        # get job name
        name = os.environ['name']
        # get slurm_job_id
        if workdir and name:
            info = load_taskinfo(workdir)
            job_names = list(
                filter(lambda j: j['name'] in [name], info['jobs']))
            if len(job_names) > 0:
                job_info = job_names[0]
                try:
                    slurm_job_id = int(job_info['slurm_job_id'])
                except Exception:
                    slurm_job_id = None
        _, _, status = callback_utils.get_slurm_job_id()
        if job_pid and job_log_path and workdir and name and slurm_job_id and status and status == 'R':
            print('slurm_job_status: R')
            sys.stdout.flush()
            break
        # break if job_pid is die.
        if job_pid and (not psutil.pid_exists(job_pid)):
            break

    # monitor log
    last_lines_hash = None
    last_lines_hash_start_time = time.time()
    time_limited_start_time = None
    while True:
        if (not job_pid or
            not job_log_path or
            not workdir or
            not name or
                not slurm_job_id):
            break
        time.sleep(60)
        is_time_limit = False
        # get last some lines
        log_lines = read_log_last(job_log_path, last_line_num=10)
        if log_lines is not None:
            try:
                log_lines = [str(line, encoding="utf-8") for line in log_lines]
            except Exception:
                log_lines = ['None']

        # get log hash
        lines_hash = get_hash(log_lines)
        # monitor whether the log has not changed over time (kill all process if not change for a long time)
        if lines_hash == last_lines_hash:
            if time.time() - last_lines_hash_start_time >= callback_common.wait_time_log_no_change * 60 * 60:
                kill_task(workdir, [name])
                is_time_limit = True
                if logger:
                    logger.error("Job({})[pid: {}, slurm: {}] is killed because the log has not changed for {} hours.".format(
                        name, job_pid, slurm_job_id, callback_common.wait_time_log_no_change))
        else:
            last_lines_hash = lines_hash
            last_lines_hash_start_time = time.time()
        # monitor whether a 'time limit exceeded' has occurred
        if (log_lines is not None) and (not is_time_limit):
            is_time_limit_occur = False
            for line in log_lines:
                if time_limited_flag in line:
                    is_time_limit_occur = True
                    break
            if is_time_limit_occur and time_limited_start_time is not None:
                if time.time() - time_limited_start_time >= callback_common.wait_time_occur_time_limited * 60 * 60:
                    kill_task(workdir, [name])
                    is_time_limit = True
                    if logger:
                        logger.error("Job({})[pid: {}, slurm: {}] is killed because the log occurs '{}' for {} hours.".format(
                            name, job_pid, slurm_job_id, time_limited_flag, callback_common.wait_time_occur_time_limited))
            else:
                time_limited_start_time = time.time()

        # break if occur '[E] Time limit exceeded'
        if is_time_limit:
            break
        # break if job_pid is die.
        if not psutil.pid_exists(job_pid):
            break

    if logger:
        logger.info("Job({})[pid: {}, slurm: {}] The child process monitoring the log has exited, \
                      with [job_pid: {}, job_log_path: {}, workdir: {}, name: {}, slurm_job_id: {}]".format(
            name, job_pid, slurm_job_id, job_pid, job_log_path, workdir, name, slurm_job_id))
