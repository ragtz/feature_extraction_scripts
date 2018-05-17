#!/usr/bin/env python

from keyframe_labels_utils import *
import numpy as np
import pickle
import argparse

OR = lambda z: reduce(lambda x, y: x or y, z)

def compile_label_disagreements(c1, c2, pregrasp):
    check_combatability(c1, c2)

    d = {}

    for pid in c1:
        for task in c1[pid]:
            for demo in c1[pid][task]:
                if not pregrasp:
                    c1[pid][task][demo] = relabel(c1[pid][task][demo], task)
                    c2[pid][task][demo] = relabel(c2[pid][task][demo], task)

                if not pid in d:
                    d[pid] = {}

                if not task in d[pid]:
                    d[pid][task] = {}

                demo_d = np.array(c1[pid][task][demo]) != np.array(c2[pid][task][demo])
                d[pid][task][demo] = list(np.where(demo_d)[0])

    return d

def print_disagreements(d):
    pids = np.array(d.keys())
    pids = pids[np.argsort([int(pid[1:]) for pid in pids])]

    for pid in pids:
    
        print '=============================='
        print 'PID:', pid
        print '=============================='

        # if at least one demo among all tasks in pid as disagreements
        if OR([d[pid][task][demo] for task in d[pid] for demo in d[pid][task]]):
            for task in d[pid]:
                # if at least one demo in task has disagreements
                if OR([d[pid][task][demo] for demo in d[pid][task]]):

                    print 'Task:', task

                    demos = np.array(d[pid][task].keys())
                    demos = demos[np.argsort([int(demo) for demo in demos])]

                    for demo in demos:
                        # if demo has at least one disagreement
                        if d[pid][task][demo]:
                            print '    - Demo:', demo

                            for kf in d[pid][task][demo]:
                                print '        +', kf
        else:
            print 'Done.'

def main():
    parser = argparse.ArgumentParser(description='Print points of disagreement between coders')
    parser.add_argument('--c1', metavar='PKL', required=True, help='Lables assigned by first coder')
    parser.add_argument('--c2', metavar='PKL', required=True, help='Labels assigned by second coder')    
    parser.add_argument('--pregrasp', action='store_true', help='Include pre/post grasp labels')

    args = parser.parse_args()
    c1 = pickle.load(open(args.c1))
    c2 = pickle.load(open(args.c2))
    pregrasp = args.pregrasp
    
    d = compile_label_disagreements(c1, c2, pregrasp)
    print_disagreements(d)

if __name__ == '__main__':
    main()

