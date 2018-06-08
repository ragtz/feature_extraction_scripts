#!/usr/bin/env python

from fetch_files import *
import collections
import rosbag
import yaml
import csv

def increment_demos(d, pid, task):
    if not pid in d:
        d[pid] = {}

    if not task in d[pid]:
        d[pid][task] = 0

    d[pid][task] += 1

def check_compatibility(bag_dir, steps_file, tasks=[]):
    bag_files = sort_by_timestamp(get_files_recursive(bag_dir, dirs_get=tasks, type='bag'))
    n = len(bag_files)

    # get num demos per pid/task from bag directory
    bag_demos = {}
    for bag_file in bag_files:
        pid = get_task_name(bag_file)
        task = get_skill_name(bag_file)
        increment_demos(bag_demos, pid, task)

    # get num demos per pid/task from steps csv file
    csv_demos = {}
    with open(steps_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pid = 'p'+'{:0>2d}'.format(int(row['PID']))
            task = row['Task']
            if task in tasks:
                increment_demos(csv_demos, pid, task)

    # iterate through bag and csv num demos, throw exception if inconsistent
    # check that pids match
    bag_pids = bag_demos.keys()
    csv_pids = csv_demos.keys()
    if bag_pids != csv_pids:
        raise Exception('PIDs differ between bag directory and steps file')

    # check that tasks per pid match
    for pid in bag_demos:
        bag_tasks = bag_demos[pid].keys()
        csv_tasks = csv_demos[pid].keys()

        if bag_tasks != csv_tasks:
            raise Exception('Tasks differ between bag directory and steps file for pid ' + pid)

    # check that num demos per pid/task match
    for pid in bag_demos:
        for task in bag_demos[pid]:
            bag_n = bag_demos[pid][task]
            csv_n = csv_demos[pid][task]

            if bag_n != csv_n:
                raise Exception('Number of demos differ between bag directory and steps file for pid ' + pid + ' and task ' + task)

def get_steps(steps_file):
    steps = {}

    with open(steps_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pid = 'p'+'{:0>2d}'.format(int(row['PID']))
            task = row['Task']
            demo_num = int(row['Demo Num'])
            step_points = np.array([float(x) for x in row['Step Points'][1:-1].split(',')])

            if not pid in steps:
                steps[pid] = {}

            if not task in steps[pid]:
                steps[pid][task] = {}

            steps[pid][task][demo_num] = step_points

    return steps

def get_demo_dict(steps_file):
    demos = {}
    
    with open(steps_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pid = 'p'+'{:0>2d}'.format(int(row['PID']))
            task = row['Task']
            demo_num = int(row['Demo Num'])

            if not pid in demos:
                demos[pid] = {}

            if not task in demos[pid]:
                demos[pid][task] = []

            demos[pid][task].append(demo_num)

    for pid in demos:
        for task in demos[pid]:
            demos[pid][task] = sorted(demos[pid][task])

    return demos

def get_bag_start(bag_file):
    with rosbag.Bag(bag_file) as bag:
        bag_info = yaml.load(bag._get_yaml_info())
        start = bag_info['start']
    return start

def adjust_time_to_ref(start_ref, start_query, t_query):
    def f(x):
        return x + (start_query - start_ref)

    if isinstance(t_query, collections.Iterable):
        return np.array([f(t) for t in t_query])
    else:
        return f(t_query)

def adjust_kf_to_ref(ref_bag, query_bag, kf):
    start_ref = get_bag_start(ref_bag)
    start_query = get_bag_start(query_bag)
    return adjust_time_to_ref(start_ref, start_query, kf)

