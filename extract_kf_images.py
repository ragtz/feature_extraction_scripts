#!/usr/bin/env python

from os.path import join, expanduser, basename
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from fetch_files import *
import numpy as np
import argparse
import rosbag
import yaml
import csv
import cv2

def resample_interp(x, t, min_t, max_t, hz):
    f = interp1d(t, x, kind='linear', axis=0)
    t_rs = np.arange(min_t, max_t, 1.0/hz)
    x_rs = f(t_rs)
    return x_rs, t_rs

def smooth(x):
    b, a = butter(2, 0.125)
    return filtfilt(b, a, x, axis=0, method='gust')

def get_images(bag_file, compressed=True, img_topic='/kinect/qhd/image_color_rect/compressed'):
    bridge = CvBridge()

    with rosbag.Bag(bag_file) as bag:
        bag_info = yaml.load(bag._get_yaml_info())
        start_t = bag_info['start']

        imgs = []
        img_t = []

        for topic, msg, t in bag.read_messages(topics=img_topic):
            t = t.to_sec()

            if compressed:
                imgs.append(bridge.compressed_imgmsg_to_cv2(msg))
            else:
                imgs.append(bridge.imgmsg_to_cv2(msg, 'bgr8'))

            img_t.append(t)

    img_t = np.array(img_t)
    img_t = img_t - start_t

    return imgs, img_t

def get_x_at_t(x_ref, t_ref, t_query):
    x_query = []

    for t in t_query:
        idx = np.argmin(np.abs(t_ref - t))
        x_query.append(x_ref[idx])

    return x_query    

def get_keyframes(bag_file):
    kf_topic = '/kf_tracker/state'
    
    with rosbag.Bag(bag_file) as bag:
        bag_info = yaml.load(bag._get_yaml_info())
        start_t = bag_info['start']
        end_t = bag_info['end']

        kf = []
        kf_state = 0

        for topic, msg, t in bag.read_messages(topics=kf_topic):
            t = t.to_sec()

            if msg.data != kf_state:
                kf.append(t)
                kf_state = msg.data

    kf = np.array(kf)
    kf = kf - start_t
    kf = [0] + list(kf) + [end_t - start_t]

    return kf

def get_gripper_keyframes(bag_file):
    gripper_state_topic = '/vector/right_gripper/stat'

    with rosbag.Bag(bag_file) as bag:
        bag_info = yaml.load(bag._get_yaml_info())
        start_t = bag_info['start']

        g_state = []
        g_t = []

        for topic, msg, t in bag.read_messages(topics=gripper_state_topic):
            t = t.to_sec()

            g_state.append(msg.position)
            g_t.append(t)

    g_t = np.array(g_t)
    g_t = g_t - start_t
        
    g_state, g_t = resample_interp(g_state, g_t, np.min(g_t), np.max(g_t), 10)
    g_state = smooth(g_state)

    d_g_state = (g_state[1:] - g_state[:-1]) / (g_t[1:] - g_t[:-1])
    g_change = map(lambda x: x > 0.0025, np.abs(d_g_state))
    g_change = map(lambda x: x[0] != x[1], zip(g_change[:-1], g_change[1:]))
    g_change_idxs = [i for i, x in enumerate(g_change) if x]

    g_kf = g_t[g_change_idxs]

    return list(g_kf)

def get_keyframe_images(bag_file):
    imgs, img_t = get_images(bag_file)
    kf = get_keyframes(bag_file)
    g_kf = get_gripper_keyframes(bag_file)

    kf = kf + g_kf
    kf = np.sort(kf)

    kf_imgs = get_x_at_t(imgs, img_t, kf)

    '''
    kf_imgs = []
    for t in kf:
        idx = np.argmin(np.abs(img_t - t))
        kf_imgs.append(imgs[idx])
    '''

    return kf_imgs, kf

def increment_demos(d, pid, task):
    if not pid in d:
        d[pid] = {}

    if not task in d[pid]:
        d[pid][task] = 0

    d[pid][task] += 1

def check_compatibility(bag_dir, steps_file):
    bag_files = sort_by_timestamp(get_files_recursive(bag_dir, type='bag'))
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

def main():
    parser = argparse.ArgumentParser(description='Extract images at keyframes')
    parser.add_argument('--robot_src', metavar='DIR', required=True, help='Path to directory containing robot bag files')
    parser.add_argument('--vid_src', metavar='DIR', required=True, help='Path to directory containing third-person video bag files')
    parser.add_argument('--target', metavar='DIR', required=True, help='Path to directory for saved images')
    parser.add_argument('--steps', metavar='CSV', required=True, help='Name of csv steps file')

    args = parser.parse_args()
    r_src = args.robot_src
    v_src = args.vid_src
    tar = args.target
    steps = args.steps

    check_compatibility(r_src, steps)
    check_compatibility(v_src, steps)

    demos = get_demo_dict(steps)
    demos_cnt = {}
    for pid in demos:
        for task in demos[pid]:
            if not pid in demos_cnt:
                demos_cnt[pid] = {}

            if not task in demos_cnt[pid]:
                demos_cnt[pid][task] = 0

    r_bag_files = sort_by_timestamp(get_files_recursive(r_src, type='bag'))
    v_bag_files = sort_by_timestamp(get_files_recursive(v_src, type='bag'))

    n = len(r_bag_files)

    for i, bag_file in enumerate(r_bag_files):
        print 'Processing ' + bag_file + ' (' + str(i) + '/' + str(n) + ')...'

        pid = get_task_name(bag_file)
        task = get_skill_name(bag_file)
        demo_idx = demos_cnt[pid][task]
        demo_id = 'd'+str(demos[pid][task][demo_idx])+'_'+get_timestamp(bag_file)

        r_imgs, t = get_keyframe_images(bag_file)
        
        v_imgs, v_t = get_images(v_bag_files[i], compressed=False, img_topic='/usb_cam/image_raw')
        v_imgs = get_x_at_t(v_imgs, v_t, t)

        for j, imgs in enumerate(zip(r_imgs, v_imgs)):
            r_img, v_img = imgs
            img_name = 'kf_'+str(j)+'_'+str(float('{0:.3f}'.format(t[j])))+'.png'
            
            r_filename = join(expanduser('~'), tar, 'robot', pid, task, demo_id, img_name)
            v_filename = join(expanduser('~'), tar, 'vid', pid, task, demo_id, img_name)
            
            ensure_dir(r_filename)
            cv2.imwrite(r_filename, r_img)
            
            ensure_dir(v_filename)
            cv2.imwrite(v_filename, v_img) 

        demos_cnt[pid][task] += 1

if __name__ == '__main__':
    main()

