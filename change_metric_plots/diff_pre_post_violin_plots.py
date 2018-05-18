#!/usr/bin/env python

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

def get_pid_num(pid):
    try:
        return int(pid)
    except:
        return int(pid[1:])

def get_did_num(did):
    try:
        return int(did)
    except:
        return int(did.split('_')[0][1:])

def sort_keys(keys, f):
    return [f(k) for k in np.array(keys)[np.argsort([f(k) for k in keys])]]

def sort_pids(pids):
    return sort_keys(pids, get_pid_num)

def sort_dids(dids):
    return sort_keys(dids, get_did_num)

def handle_pids(data, pids):
    if not pids:
        pids = list(set(data.keys()))
    return pids

def handle_tasks(data, pids, tasks):
    if not tasks:
        tasks = set([])
        for pid in pids:
            tasks = tasks.union(data[pid].keys())
        tasks = list(tasks)
    return tasks    

def check_compatability(data, labels):
    d1, d2 = data, labels

    # check that pids match
    d1_pids = sort_pids(d1.keys())
    d2_pids = sort_pids(d2.keys())

    if d1_pids != d2_pids:
        raise Exception('PIDs differ between data sources')

    # check that tasks per pid match
    for pid in d1:
        d1_tasks = sorted(d1[pid].keys())
        d2_tasks = sorted(d2[pid].keys())

        if d1_tasks != d2_tasks:
            raise Exception('Tasks differ between data sources for pid ' + pid)

    # check that demos per pid/task match
    for pid in d1:
        for task in d1[pid]:
            d1_demos = sort_dids(d1[pid][task].keys())
            d2_demos = sort_dids(d2[pid][task].keys())

            if d1_demos != d2_demos:
                raise Exception('Demos differ between data sources for pid ' + pid + ' and task ' + task)

    # check that num keyframes per pid/task/demo match
    for pid in d1:
        for task in d1[pid]:
            for demo in d1[pid][task]:
                d1_n = len(d1[pid][task][demo]['kf'])
                d2_n = len(d2[pid][task][str(get_did_num(demo))])

                if d1_n != d2_n:
                    raise Exception('Number of keyframes differ between data sources for pid ' + pid + ', task ' + task + ', and demo ' + demo)

def data_subset(data, pids, tasks):
    pids = handle_pids(data, pids)
    tasks = handle_tasks(data, pids, tasks)

    d = {}

    for pid in data:
        if pid in pids:
            for task in data[pid]:
                if task in tasks:
                    for demo in data[pid][task]:
                        if not pid in d:
                            d[pid] = {}

                        if not task in d[pid]:
                            d[pid][task] = {}

                        d[pid][task][demo] = deepcopy(data[pid][task][demo])

    return d 

def get_segment(d, t, t_start, t_end):
    idx_start = np.argmin(np.abs(t - t_start))
    idx_end = np.argmin(np.abs(t - t_end))
    return d[idx_start:idx_end]

def compute_statistics(traj):
    return {'mean': np.mean(traj, axis=0),
            'median': np.median(traj, axis=0),
            'variance': np.var(traj, axis=0)}

def compute_diff_statistics(pre, post):
    return {'mean_diff': np.abs(np.mean(post, axis=0) - np.mean(pre, axis=0)),
            'median_diff': np.abs(np.median(post, axis=0) - np.median(pre, axis=0)),
            'variance_diff': np.abs(np.var(post, axis=0) - np.var(pre, axis=0))}

def compile_statistics(demo, labels, delta_t, stats):
    t = demo['t']
    keyframes = demo['kf']
    state = demo['state']
    state_names = demo['state_names']

    for name in state_names:
        if not name in stats:
            stats[name] = [{'pre': {}, 'post': {}},
                           {'pre': {}, 'post': {}}]

    for i, kf in enumerate(keyframes[1:-1]):
        i += 1

        pre = get_segment(state, t, kf-delta_t, kf)
        post = get_segment(state, t, kf, kf+delta_t)

        if len(pre) > 4 and len(post) > 4:
            pre_stats = compute_statistics(pre)
            post_stats = compute_statistics(post)

            for j, name in enumerate(state_names):
                pre = stats[name][labels[i]]['pre']
                post = stats[name][labels[i]]['post']

                for k in pre_stats:
                    if not k in pre:
                        pre[k] = []

                    pre[k].append(pre_stats[k][j])
                
                for k in post_stats:
                    if not k in post:
                        post[k] = []

                    post[k].append(post_stats[k][j])

def compile_diff_statistics(demo, labels, delta_t, stats):
    t = demo['t']
    keyframes = demo['kf']
    state = demo['state']
    state_names = demo['state_names']

    for name in state_names:
        if not name in stats:
            stats[name] = [{}, {}]

    for i, kf in enumerate(keyframes[1:-1]):
        i += 1

        pre = get_segment(state, t, kf-delta_t, kf)
        post = get_segment(state, t, kf, kf+delta_t)

        if len(pre) > 4 and len(post) > 4:
            diff = compute_diff_statistics(pre, post)

            for j, name in enumerate(state_names):
                s = stats[name][labels[i]]

                for k in diff:
                    if not k in s:
                        s[k] = []

                    s[k].append(diff[k][j])
 
def plot_statistics(stats, fname, stat='median', n_cols=4):
    n = len(stats)
    n_rows = n/n_cols + (n%n_cols > 0)

    fig, ax = plt.subplots(n_rows, n_cols, sharex=True)#, sharey=True)
    for idx, name in enumerate(sorted(stats.keys())):
        pos = [0, 1, 3, 4]
        labels = ['ns-pre', 'ns-post', 's-pre', 's-post']

        ns_pre = stats[name][0]['pre'][stat]
        ns_post = stats[name][0]['post'][stat]

        s_pre = stats[name][1]['pre'][stat]
        s_post = stats[name][1]['post'][stat]

        data = [ns_pre, ns_post, s_pre, s_post]

        i, j = idx/n_cols, idx%n_cols

        ax[i,j].set_title(name)
        ax[i,j].violinplot(data, pos, showmeans=True, showextrema=True, showmedians=True)
        ax[i,j].set_xticks(pos)
        ax[i,j].set_xticklabels(labels)
        
        if i == n_rows-1:
            ax[i,j].set_xlabel('Label')
        
        if j == 0:
            ax[i,j].set_ylabel(stat)

    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()

def plot_diff_statistics(stats, fname, task, stat='median_diff', n_cols=4):
    n = len(stats)
    n_rows = n/n_cols + (n%n_cols > 0)

    fig, ax = plt.subplots(n_rows, n_cols, sharex=True)#, sharey=True)
    fig.suptitle(task)
    for idx, name in enumerate(sorted(stats.keys())):
        pos = [0, 1]
        labels = ['non-seg', 'seg']

        data_0 = stats[name][0][stat]
        data_1 = stats[name][1][stat]
        data = [data_0, data_1]

        i, j = idx/n_cols, idx%n_cols

        ax[i,j].set_title(name)
        ax[i,j].violinplot(data, pos, showmeans=True, showextrema=True, showmedians=True)
        ax[i,j].set_xticks(pos)
        ax[i,j].set_xticklabels(labels)
        
        if i == n_rows-1:
            ax[i,j].set_xlabel('Label')
        
        if j == 0:
            ax[i,j].set_ylabel(stat)

    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
     
def main():
    parser = argparse.ArgumentParser(description='Segment keyframe demonstrations')
    parser.add_argument('--data', metavar='PKL', required=True, help='Dataset pkl file')
    parser.add_argument('--labels', metavar='PKL', required=True, help='Keyframe labels pkl file')
    parser.add_argument('--delta_t', metavar='SEC', type = float, required=True, help='Time window before and after keyframe') 
    parser.add_argument('--pids', metavar='PID', nargs='+', default=[], required=False, help='Pids from dataset to process')
    parser.add_argument('--tasks', metavar='TASK', nargs='+', default=[], required=False, help='Tasks from dataset to process')
    parser.add_argument('--fname', metavar='STR', required=False, help='Name of saved figure')

    args = parser.parse_args()
    data = pickle.load(open(args.data))
    labels = pickle.load(open(args.labels))
    delta_t = args.delta_t
    pids = args.pids
    tasks = args.tasks
    fname = args.fname

    data = data_subset(data, pids, tasks)
    labels = data_subset(labels, pids, tasks)

    check_compatability(data, labels)

    stats = {}
    for pid in data:
        for task in data[pid]:
            for demo in data[pid][task]:
                compile_diff_statistics(data[pid][task][demo], 
                                        labels[pid][task][str(get_did_num(demo))],
                                        delta_t,
                                        stats)

    plot_diff_statistics(stats, fname, tasks[0])
 
if __name__ == '__main__':
    main()

