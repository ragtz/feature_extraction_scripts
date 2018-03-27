#!/usr/bin/env python

from itertools import groupby
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import sys

matplotlib.rcParams.update({'font.size': 22})

def plot_vert_line(ax, idx, t, c='b', l='-', w=3, min_y=0, max_y=.1):
    return ax[idx].plot([t,t], [min_y,max_y], c=c, linestyle=l, linewidth=w)[0]

def plot_vert_lines(ax, idx, T, c='b', l='-', min_y=0, max_y=.1):
    for t in T:
        line = plot_vert_line(ax, idx, t, c=c, l=l, min_y=min_y, max_y=max_y)
    return line

def plot_seg(demos, kf_sets, step_sets, pids=None, title=''):
    if len(demos) != len(kf_sets):
        raise Exception("Number of demos must match number of keyframe sets")
    elif len(demos) != len(step_sets):
        raise Exception("Number of demos must match number of step sets")
        
    fig, ax = plt.subplots(len(demos), 1, sharex=True)
    for i, demo in enumerate(demos):
        kf_line = plot_vert_lines(ax, i, kf_sets[i], c='r')
        step_line = plot_vert_lines(ax, i, step_sets[i], c='b', l='--')
        gripper_line = ax[i].plot(demo['t'], demo['x'], c='k', linewidth=3)[0]
        if not pids is None:
            ax[i].set_title(pids[i])
        ax[i].yaxis.set_visible(False)

    fig.suptitle(title)
    #fig.legend([kf_line, step_line, gripper_line], ['Keyframe', 'Step', 'Gripper'], loc='center right')
    fig.legend([kf_line, step_line, gripper_line], ['KF', 'Step', 'Grip'], loc='center right')
    ax[-1].set_xlabel('Time (s)')

def main():
    data_file = sys.argv[1]
    target_task = sys.argv[2]
    data = pickle.load(open(data_file))

    pids = []
    demos = []
    kf = []
    step = []

    for pid in data:
        for task in data[pid]:
            if task == target_task:
                for demo_id in data[pid][task]:
                    demo = data[pid][task][demo_id]
                    g_idx = np.where(demo['state_names'] == 'gripper_state')[0]

                    grip = {'t': demo['t'],
                            'x': demo['state'][:,g_idx]}

                    pids.append(pid)
                    demos.append(grip)
                    kf.append(demo['kf'])
                    step.append(demo['steps'])

    plot_seg(demos, kf, step, pids, target_task)
    plt.show()

if __name__ == '__main__':
    main()
