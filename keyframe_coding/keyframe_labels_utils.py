#!/usr/bin/env python

remove_pre_post_grasp = {'drawer': {0: 1,
                                    5: 4},
                         'lamp': {0: 1,
                                  6: 5},
                         'pitcher': {0: 1,
                                     7: 6,
                                     8: 9,
                                     16: 15},
                          'bowl': {0: 1,
                                   10: 9}}

def relabel(labels, task):
    remap = remove_pre_post_grasp[task]
    return [remap[label] if label in remap else label for label in labels]

def check_combatability(c1, c2):
    # check that pids match
    c1_pids = sorted(c1.keys())
    c2_pids = sorted(c2.keys())
    
    if c1_pids != c2_pids:
        raise Exception('PIDs differ between two label sets')

    # check that tasks per pid match
    for pid in c1:
        c1_tasks = sorted(c1[pid].keys())
        c2_tasks = sorted(c2[pid].keys())

        if c1_tasks != c2_tasks:
            raise Exception('Tasks differ between label sets for pid ' + pid)

    # check that demos per pid/task match
    for pid in c1:
        for task in c1[pid]:
            c1_demos = sorted(c1[pid][task].keys())
            c2_demos = sorted(c2[pid][task].keys())

            if c1_demos != c2_demos:
                raise Exception('Demos differ between label sets for pid ' + pid + 'and task ' + task)

    # check that num keyframes per pid/task/demo match
    for pid in c1:
        for task in c1[pid]:
            for demo in c1[pid][task]:
                c1_n = len(c1[pid][task][demo])
                c2_n = len(c2[pid][task][demo])

                if c1_n != c2_n:
                    raise Exception('Number of keyframes differ between label sets for pid ' + pid + ', task ' + task + ', and demo ' + demo)

