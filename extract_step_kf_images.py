#!/usr/bin/env python

from os.path import join, expanduser, basename
from extract_kf_images import *
from demo_file_utils import *
from fetch_files import *
import numpy as np
import argparse
import rosbag
import csv
import cv2

def main():
    parser = argparse.ArgumentParser(description='Extract step and nearest keyframe images')
    parser.add_argument('--robot_src', metavar='DIR', required=True, help='Path to directory containing robot bag files')
    parser.add_argument('--vid_src', metavar='DIR', required=True, help='Path to directory containing third-person video bag files')
    parser.add_argument('--target', metavar='DIR', required=True, help='Path to directory for saved images')
    parser.add_argument('--steps', metavar='CSV', required=True, help='Path to csv steps file')

    args = parser.parse_args()
    r_src = args.robot_src
    v_src = args.vid_src
    tar = args.target
    steps = args.steps

    check_compatibility(r_src, steps)
    check_compatibility(v_src, steps)

    steps = get_steps(steps)
    demos_cnt = {}
    for pid in steps:
        for task in steps[pid]:
            if not pid in demos_cnt:
                demos_cnt[pid] = {}

            if not task in demos_cnt[pid]:
                demos_cnt[pid][task] = 0

    r_bag_files = sort_by_timestamp(get_files_recursive(r_src, type='bag'))
    v_bag_files = sort_by_timestamp(get_files_recursive(v_src, type='bag'))

    n = len(r_bag_files)

    for i, bag_files in enumerate(zip(r_bag_files, v_bag_file)):
        r_bag_file, v_bag_file = bag_files
        print 'Processing ' + r_bag_file + '(' + str(i+1) + '/' + str(n) + ')...'

        pid = get_task_name(r_bag_file)
        task = get_skill_name(v_bag_file)

        demo_idx = demos_cnt[pid][task]
        demo_num = steps[pid][task].keys()[demo_idx]
        demo_id = 'd'+str(demo_num)+'_'+get_timestamp(r_bag_file)

        step_t = steps[pid][task][demo_num] # in vid time

        v_imgs, v_t = get_images(v_bag_file, compressed=False, img_topic='/usb_cam/image_raw')
        step_imgs = get_x_at_t(v_imgs, v_t, step_t)

        kf = get_all_keyframes(r_bag_file) # in robot time
        kf = adjust_kf_to_ref(v_bag_file, r_bag_file, kf) # in vid time

        kf_imgs = get_x_at_t(v_imgs, v_t, kf)
        kf_imgs = get_x_at_t(kf_imgs, kf, step_t)

        for j, imgs in enumerate(zip(step_imgs, kf_imgs)):
            step_img, kf_img = imgs
            step_dir = 'step_'+str(j)+'_'+str(float('{0:.3f}'.format(step_t[j])))

            #step_filename = join(expanduser('~'), tar, pid, task, demo_id, step_dir, 'step.png')
            #kf_filename = join(expanduser('~'), tar, pid, task, demo_id, step_dir, 'kf.png')
            step_filename = join(expanduser('~'), tar, pid, task, demo_id, step_dir+'.png')
            kf_filename = join(expanduser('~'), tar, pid, task, demo_id, step_dir+'_kf.png')

            ensure_dir(step_filename)
            cv2.imwrite(step_filename, step_img)

            ensure_dir(kf_filename)
            cv2.imwrite(kf_filename, kf_img)

        demos_cnt[pid][task] += 1
        
if __name__ == '__main__':
    main()

