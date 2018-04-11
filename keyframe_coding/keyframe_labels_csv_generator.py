#!/usr/bin/env python

from os.path import isfile, isdir, join, expanduser, splitext, basename
from os import listdir
import numpy as np
import argparse
import csv
import re

pid_pattern = 'p[0-9]+'
tasks = ['drawer', 'lamp', 'pitcher', 'bowl']
demo_pattern = 'd[0-9]+_2018-[0-9]{2}-[0-9]{2}T[0-9]{6}'
kf_pattern = 'kf_[0-9]+_[0-9]+.[0-9]{3}.png'

is_pid = lambda f: isdir(f) and re.match(pid_pattern, basename(f))
is_task = lambda f: isdir(f) and basename(f) in tasks
is_demo = lambda f: isdir(f) and re.match(demo_pattern, basename(f))
is_kf = lambda f: isfile(f) and re.match(kf_pattern, basename(f))

def get_demo_num(demo_str):
    return int(re.split('_', demo_str)[0][1:])

def generate_keyframe_label_csv(path):
    with open(join(path, basename(path)+'_labels.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['task', 'demo', 'kf', 'label'])
        writer.writeheader()

        for task in listdir(path):
            task_path = join(path, task)
            if is_task(task_path):
                for demo in listdir(task_path):
                    demo_path = join(task_path, demo)
                    if is_demo(demo_path):
                        #num_kfs = len([kf for kf in listdir(demo_path) if is_kf(join(demo_path, kf))])
                        num_kfs = len(listdir(demo_path))
                        writer.writerow({'task': task, 'demo': get_demo_num(demo), 'kf': 0, 'label': ''})
                        for kf in range(1,num_kfs):
                            writer.writerow({'task': '', 'demo': '', 'kf': kf, 'label': ''})

def main():
    parser = argparse.ArgumentParser(description='Generate keyframe label csv file')
    parser.add_argument('--img', metavar='DIR', required=True, help='Path to directory containing keyframe images')
    
    args = parser.parse_args()
    img_dir = args.img

    for pid in listdir(img_dir):
        if is_pid(join(img_dir, pid)):
            generate_keyframe_label_csv(join(img_dir, pid))
 
if __name__ == '__main__':
    main()

