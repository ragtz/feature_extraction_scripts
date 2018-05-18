#!/usr/bin/env python

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

colors = {0: '#7e1e9c', #'xkcd:purple',
          1: '#15b01a', #'xkcd:green',
          2: '#0343df', #'xkcd:blue',
          3: '#ff81c0', #'xkcd:pink',
          4: '#653700', #'xkcd:brown',
          5: '#e50000', #'xkcd:red',
          6: '#95d0fc', #'xkcd:light blue',
          7: '#029386', #'xkcd:teal',
          8: '#f97306', #'xkcd:orange',
          9: '#ffff14', #'xkcd:yellow',
          10: '#929591', #'xkcd:grey',
          11: '#89fe05', #'xkcd:lime green',
          12: '#6e750e', #'xkcd:olive',
          13: '#c04e01', #'xkcd:burnt orange',
          14: '#ae7181', #'xkcd:muave',
          15: '#00ffff', #'xkcd:cyan',
          16: '#13eac9', #'xkcd:aqua',
          17: '#cea2fd'} #'xkcd:lilac'}

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
    return deepcopy(t[idx_start:idx_end]), deepcopy(d[idx_start:idx_end])

def get_title(pid, task, did):
    return pid + '_' + task + '_' + did.split('_')[0]

def plot_vert_line(ax, idx, t, c='b', w=3):
    min_y, max_y = ax[idx].get_ylim()
    return ax[idx].plot([t,t], [min_y,max_y], c=c, linewidth=w)[0]

def plot_vert_lines(ax, idx, T, c='b', w=3):
    for t in T:
        line = plot_vert_line(ax, idx, t, c=c, w=w)
    return line

def compile_segments(demo, labels, prim_labels, delta_t, segs):
    t = demo['t']
    keyframes = demo['kf']
    state = demo['state']
    state_names = demo['state_names']

    for name in state_names:
        if not name in segs:
            segs[name] = [{'t': [], 'x': [], 'p': []},
                          {'t': [], 'x': [], 'p': []}]

    for i, kf in enumerate(keyframes[1:-1]):
        i += 1

        seg_t, seg_x = get_segment(state, t, kf-delta_t, kf+delta_t)
        seg_t -= kf

        if len(seg_x) > 9:
            for j, name in enumerate(state_names):
                segs[name][labels[i]]['t'].append(seg_t)
                segs[name][labels[i]]['x'].append(seg_x[:,j])
                segs[name][labels[i]]['p'].append(prim_labels[i])

def plot_overlay_time_series(segs, delta_t, fname):
    n = len(segs)

    for idx, name in enumerate(sorted(segs.keys())):
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    
        t_0 = segs[name][0]['t']
        t_1 = segs[name][1]['t']

        x_0 = segs[name][0]['x']
        x_1 = segs[name][1]['x']

        p_0 = segs[name][0]['p']
        p_1 = segs[name][1]['p']
 
        ax[0].set_title(name + ' non-seg')
        for j in range(len(t_0)):
            idx = np.argmin(np.abs(t_0[j]))
            ax[0].plot(t_0[j], x_0[j]-x_0[j][idx], c=colors[p_0[j]], alpha=0.25, lw=2)
        ax[0].set_xlim([-delta_t, delta_t])

        ax[1].set_title(name + ' seg')
        for j in range(len(t_1)):
            idx = np.argmin(np.abs(t_1[j]))
            ax[1].plot(t_1[j], x_1[j]-x_1[j][idx], c=colors[p_0[j]], alpha=0.25, lw=2)
        ax[1].set_xlim([-delta_t, delta_t])

        ax[0].set_ylim(ax[0].get_ylim())
        ax[1].set_ylim(ax[0].get_ylim())        
        plot_vert_line(ax, 0, 0, c='k', w=2)
        plot_vert_line(ax, 1, 0, c='r', w=2)

        if fname:
            plt.savefig(fname+'_'+name+'.png', bbox_inches='tight')

    if not fname:
        plt.show()
    
def main():
    parser = argparse.ArgumentParser(description='Segment keyframe demonstrations')
    parser.add_argument('--data', metavar='PKL', required=True, help='Dataset pkl file')
    parser.add_argument('--labels', metavar='PKL', required=True, help='Keyframe labels pkl file')
    parser.add_argument('--prim_labels', metavar='PKL', required=True, help='Keyframe primitive labels pkl file')
    parser.add_argument('--delta_t', metavar='SEC', type = float, required=True, help='Time window before and after keyframe') 
    parser.add_argument('--pids', metavar='PID', nargs='+', default=[], required=False, help='Pids from dataset to process')
    parser.add_argument('--tasks', metavar='TASK', nargs='+', default=[], required=False, help='Tasks from dataset to process')
    parser.add_argument('--fname', metavar='STR', required=False, help='Name of saved figure')

    args = parser.parse_args()
    data = pickle.load(open(args.data))
    labels = pickle.load(open(args.labels))
    prim_labels = pickle.load(open(args.prim_labels))
    delta_t = args.delta_t
    pids = args.pids
    tasks = args.tasks
    fname = args.fname

    data = data_subset(data, pids, tasks)
    labels = data_subset(labels, pids, tasks)
    prim_labels = data_subset(prim_labels, pids, tasks)

    check_compatability(data, labels)
    check_compatability(data, prim_labels)

    segs = {}
    for pid in data:
        for task in data[pid]:
            for demo in data[pid][task]:
                compile_segments(data[pid][task][demo], 
                                 labels[pid][task][str(get_did_num(demo))],
                                 prim_labels[pid][task][str(get_did_num(demo))],
                                 delta_t,
                                 segs)

    plot_overlay_time_series(segs, delta_t, fname)
 
if __name__ == '__main__':
    main()

