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

def change(pre, post):
    if len(pre) >= 5 and len(post) >= 5:
        joint = np.concatenate((pre,post))
        var_pre = np.var(pre)
        var_post = np.var(post)
        var_joint = np.var(joint)

        var_pre_joint_ratio = var_pre/var_joint
        var_post_joint_ratio = var_post/var_joint
        
        return np.max([var_pre_joint_ratio, var_post_joint_ratio])
    else:
        return 0

def get_change_values(x, t, delta_t):
    c = []
    for t_i in t:
        pre = get_segment(x, t, t_i-delta_t, t_i)
        post = get_segment(x, t, t_i, t_i+delta_t)
        c.append(change(pre, post))
    return np.array(c)

def get_title(pid, task, did):
    return pid + '_' + task + '_' + did.split('_')[0]

def plot_vert_line(ax, idx, t, c='b', w=3, min_y=-1, max_y=1):
    return ax[idx].plot([t,t], [min_y,max_y], c=c, linewidth=w)[0]

def plot_vert_lines(ax, idx, T, c='b', w=3, min_y=-1, max_y=1):
    for t in T:
        line = plot_vert_line(ax, idx, t, c=c, w=w, min_y=min_y, max_y=max_y)
    return line

def plot_change_values(title, demo, labels, delta_t, fname):
    kf_colors = ['k', 'r']

    t = demo['t']
    keyframes = np.array(demo['kf'])
    state = demo['state']
    state_names = demo['state_names']
    labels = np.array(labels)

    n = len(state_names)
    fig, ax = plt.subplots(n, 1, sharex=True)

    for i, name in enumerate(state_names):
        s = state[:,i]
        c_values = get_change_values(s, t, delta_t)

        s = s / max([abs(min(s)), abs(max(s))])
        c_values = c_values / max([abs(min(c_values)), abs(max(c_values))])
        
        ax[i].set_title(name)
        f_line = ax[i].plot(t, s, c='b')[0]
        c_line = ax[i].plot(t, c_values, c='g')[0]
        
        ns_kfs = keyframes[np.where(labels == 0)]
        ns_line = plot_vert_lines(ax, i, ns_kfs, c=kf_colors[0])

        s_kfs = keyframes[np.where(labels == 1)]
        s_line = plot_vert_lines(ax, i, s_kfs, c=kf_colors[1])

    fig.suptitle(title)
    fig.legend([f_line, c_line, ns_line, s_line], ['Feature', 'C-Value', 'Non-Seg', 'Seg'], loc='center right')
    ax[-1].set_xlabel('Time (s)')

    if fname:
        plt.savefig(fname+'/'+title+'.png', bbox_inches='tight')
    
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

    for pid in data:
        for task in data[pid]:
            for demo in data[pid][task]:
                plot_change_values(get_title(pid, task, demo),
                                   data[pid][task][demo],
                                   labels[pid][task][str(get_did_num(demo))],
                                   delta_t, fname)
    if not fname:    
        plt.show()
 
if __name__ == '__main__':
    main()

