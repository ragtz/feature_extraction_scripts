#!/usr/bin/env python

from geometry_msgs.msg import PointStamped
from os.path import join, expanduser, splitext, basename
from scipy.signal import butter, filtfilt, medfilt
from scipy.interpolate import interp1d
from tf.transformations import compose_matrix
from tf import TransformerROS
from yaml_include_loader.loader import *
from object_filters import *
from demo_file_utils import *
from fetch_files import *
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import itertools
import argparse
import colorsys
import pickle
import rosbag
import yaml
import csv
import re

import matplotlib.pyplot as plt

resample = None

def dist(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

def z_dist(x, y):
    return abs(x[2] - y[2])

def make_np(d):
    for k in d:
        d[k] = np.array(d[k])

def transformPoint(p, header, tf, frame='/base_link'):
    return np.dot(tf, list(p)+[1])[:3]
    '''
    ps = PointStamped()
    ps.header = header
    ps.point.x = p[0]
    ps.point.y = p[1]
    ps.point.z = p[2]

    p_tf = tf.transformPoint(frame, ps)
    return [p_tf.point.x, p_tf.point.y, p_tf.point.z]
    '''

# Input: steps csv file
#        demo pid
#        task name
# Returns: all ordered indices of demos with pid, task
def get_step_demo_idxs(steps_file, pid, task):
    demo_nums = []
    idxs = []
    
    with open(steps_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if int(row['PID']) == int(pid[1:]) and row['Task'] == task:
                demo_nums.append(int(row['Demo Num']))
                idxs.append(i)

    return list(np.array(idxs)[np.argsort(demo_nums)])

# Input: ExtractedFeaturesArray msg
# Returns: PlaneFeatures msg
def get_table(msg):
    return msg.planes[0]

# Input: PlaneFeatures msg
# Returns: plane coeffs
def get_table_plane(table, tf, frame='/base_link'):
    coeffs = table.coeffs.data

    t = [0, 0, -coeffs[3]/coeffs[2]]
    t_tf = transformPoint(t, table.header, tf)

    h = [t[0] + coeffs[0], t[1] + coeffs[1], t[2] + coeffs[2]]
    h_tf = transformPoint(h, table.header, tf)

    norm_tf = np.array(h_tf) - t_tf
    norm_tf = norm_tf/np.linalg.norm(norm_tf)

    return np.array([norm_tf[0], norm_tf[1], norm_tf[2], -np.dot(norm_tf, t_tf)])

# Input: plane coeffs
# Returns: table height
def get_table_height(table):
    return -table[3]/table[2]

# Input: ExtractedFeaturesArray msg
# Returns: ObjectFeatures msg
def get_object(msg, filt):
    idx = filt(msg)
    if idx:
        return msg.objects[idx[0]]
    else:
        None

# Input: ObjectFeatures msg
# Returns: position np array
def get_object_position(obj, tf, frame='/base_link'):
    p = obj.obb.bb_center
    p = transformPoint([p.x, p.y, p.z], obj.header, tf)
    return np.array(p)

# Input: ObjectFeatures msg, PlaneFeatures msg,
#        assumes object and plane in same frame
# Returns: normal histogram np array
def get_object_histogram(obj, table):
    norms = []
    i = 0
    
    for normal in pc2.read_points(obj.normals, skip_nans=True):
        norms.append([normal[0], normal[1], normal[2]])
        i += 1

    if len(norms) > 0:
        table = table.coeffs.data
        p_norm = [table[0], table[1], table[2]]
        theta = abs(np.dot(np.array(norms), p_norm))

        hist = np.histogram(norms, 20, range=(0,1))[0]
        hist = hist/float(sum(hist))
        return np.array([hist[0], hist[len(hist)/2], hist[-1]])
    else:
        return None

# Input: ObjectFeatures msg
# Returns: object volume
def get_object_volume(obj):
    p = obj.obb.bb_dims
    return p.x*p.y*p.z

# Input: ObjectFeatures msg
# Returns: hsv np array
def get_object_color(obj):
    rgba = obj.basicInfo.rgba_color
    h, s, v = colorsys.rgb_to_hsv(rgba.r, rgba.g, rgba.b)
    return np.array([h, s, v])

# Input: Pose msg
# Returns: position np array
def get_gripper_position(msg):
    return np.array([msg.position.x, msg.position.y, msg.position.z])

# Input: GripperStat msg
# Returns: open gripper state
def get_gripper_state(msg):
    return msg.position

def get_force(msg):
    force = msg.wrench.force
    return np.array([force.x, force.y, force.z])

def get_torque(msg):
    torque = msg.wrench.torque
    return np.array([torque.x, torque.y, torque.z])

def resample_clamp(x, t, min_t, max_t, hz):
    t = np.array(t)
    t_rs = np.arange(min_t, max_t, 1./hz)
    x_rs = []

    for i in range(len(t_rs)):
        t_past = np.array(filter(lambda z: z[1] <= 0, enumerate(t - t_rs[i])))
        x_rs.append(x[int(t_past[:,0][np.argmax(t_past[:,1])])])

    return np.array(x_rs), t_rs

def resample_interp(x, t, min_t, max_t, hz):
    f = interp1d(t, x, kind='linear', axis=0)
    t_rs = np.arange(min_t, max_t, 1.0/hz)
    x_rs = f(t_rs)
    return x_rs, t_rs

def smooth(x):
    b, a = butter(2, 0.125)
    return filtfilt(b, a, x, axis=0, method='gust')
'''
def smooth(x):
    return medfilt(x)
'''

def is_hist(name):
    return '_hist' in name

def is_color(name):
    return '_color' in name

def is_volume(name):
    return '_volume' in name

def is_force(name):
    return 'force' in name

def is_torque(name):
    return 'torque' in name

def is_obj(name):
    return not is_hist(name) and not is_color(name) and not is_volume(name) and not is_force(name) and not is_torque(name)

# Input: state dictionary of form {'name': np.array([0,0,0]), ...}
#        list of objects to get histogram features
#        list of objects to get color features
# Returns: feature names, features tuple
def get_feature_vector(state, 
                       obj_hists=['pitcher','bowl','lamp','drawer','large_bowl','small_bowl','spoon'],
                       obj_colors=['lamp','drawer','bowl','pitcher', 'small_bowl', 'large_bowl', 'spoon'],
                       obj_volume=['drawer','lamp','bowl','pitcher', 'large_bowl', 'small_bowl', 'spoon']):
    table = state['table']
    gripper_position = state['gripper_position']
    gripper_state = state['gripper_state']
    force = state['force']
    torque = state['torque']
    objs = {name: state[name] for name in state if not name in ['table', 'gripper_position', 'gripper_state']}

    feature_names = []
    features = []

    # dists between objects
    for obj_1, obj_2 in itertools.combinations(objs.iteritems(), 2):
        name_1, obj_1 = obj_1
        name_2, obj_2 = obj_2
        if is_obj(name_1) and is_obj(name_2):
            feature_names.append(name_1+'_'+name_2+'_dist')
            features.append(dist(obj_1, obj_2))
    
    # dists to table
    for name, obj in objs.iteritems():
        if is_obj(name):
            feature_names.append('table_'+name+'_dist')
            features.append(z_dist([0,0,get_table_height(table)], obj))

    # dists to gripper
    for name, obj in objs.iteritems():
        if is_obj(name):
            feature_names.append('gripper_'+name+'_dist')
            features.append(dist(gripper_position, obj))
    
    # gripper state
    feature_names.append('gripper_state')
    features.append(gripper_state)

    # force
    feature_names.extend(['force_x', 'force_y', 'force_z'])
    features.extend(list(force))

    # torque
    feature_names.extend(['torque_x', 'torque_y', 'torque_z'])
    features.extend(list(torque))

    # hist features
    for name in obj_hists:
        if name+'_hist' in objs:
            hist = objs[name+'_hist']
            for i in range(len(hist)):
                feature_names.append(name+'_hist_'+str(i))
            features.extend(list(hist))

    # color features
    for name in obj_colors: 
        if name+'_color' in objs:
            color = objs[name+'_color']
            for i in range(len(color)):
                feature_names.append(name+'_color_'+str(i))
            features.extend(list(color))

    # volume features
    for name in obj_volume:
        if name+'_volume' in objs:
            volume = objs[name+'_volume']
            feature_names.append(name+'_volume')
            features.append(volume)

    return np.array(feature_names), np.array(features)

def get_state_at_idx(state, idx):
    state_slice = {}
    for k in state:
        state_slice[k] = state[k][idx]
    return state_slice

# Assumes 1-level distionary with list values
def merge_dictionaries(d1, d2):
    d_merged = {}
    for k in d1:
        d_merged[k] = d1[k] + d2[k]
    return d_merged

# Input: state time series of form {'name' : np.array([0,0,0],
#                                                     [0,0,0]), ...}
# Returns: feature names, feature vector time series of form np.array([0,0,0],
#                                                                     [0,0,0])
def get_feature_vectors(state_ts):
    n = len(state_ts[state_ts.keys()[0]])

    fv_ts = []
    for i in range(n):
        state = get_state_at_idx(state_ts, i)
        feature_names, features = get_feature_vector(state)
        fv_ts.append(features)

    return feature_names, np.array(fv_ts)

def get_action_vectors(state_ts):
    gripper = np.concatenate((state_ts['gripper_position']['x'],
                              np.array([state_ts['gripper_state']['x']]).T), axis=1)
    t = state_ts['gripper_position']['t']

    feature_names = ['gripper_x_vel', 'gripper_y_vel', 'gripper_z_vel', 'gripper_state_vel']
    features = (gripper[1:,:] - gripper[:-1,:]) / np.repeat([t[1:] - t[:-1]], 4, axis=0).T
    features = np.concatenate((features, [features[-1,:]]), axis=0)

    return np.array(feature_names), features    

# Input: name of bag file
#        dictionary of object filters
# Returns: dictionary of state time series
def extract_state_time_series(bag_file, object_filters):
    features_topic = '/beliefs/features'
    eef_topic = '/eef_pose'
    gripper_state_topic = '/vector/right_gripper/stat'
    force_topic = '/j2s7s300_driver/out/tool_wrench'

    '''
    tf = TransformerROS()
    with rosbag.Bag(bag_file) as bag:
        for _, msg, t in bag.read_messages(topics='/tf'):
            for transform in msg.transforms:
                tf.setTransform(transform) 
    '''
    # transform:
    #       source: /base_link
    #       target: /kinect_rgb_optical_frame
    tf = compose_matrix(angles=[-2.075, 0, -1.575], translate=[0.358, -0.096, 1.572])

    with rosbag.Bag(bag_file) as bag:
        bag_info = yaml.load(bag._get_yaml_info())
        start_t = bag_info['start']

        ts = {'table': {'x': [], 't': []},
              'gripper_position': {'x': [], 't': []},
              'gripper_state': {'x': [], 't': []},
              'force': {'x': [], 't': []},
              'torque': {'x': [], 't': []}}
        
        for name in object_filters:
            ts[name] = {'x': [], 't': []}
            ts[name+'_hist'] = {'x': [], 't': []}
            ts[name+'_color'] = {'x': [], 't': []}
            ts[name+'_volume'] = {'x': [], 't': []}

        for topic, msg, t in bag.read_messages(topics=[features_topic, eef_topic, gripper_state_topic, force_topic]):
            t = t.to_sec()
            if topic == features_topic:
                table = get_table(msg)
                table_plane = get_table_plane(table, tf)
                ts['table']['x'].append(table_plane)
                ts['table']['t'].append(t)

                objects = {name: get_object(msg, filt) for name, filt in object_filters.iteritems()}
                for name, obj in objects.iteritems():
                    if not obj is None:
                        object_position = get_object_position(obj, tf)
                        object_histogram = get_object_histogram(obj, table)
                        object_color = get_object_color(obj)
                        object_volume = get_object_volume(obj)

                        ts[name]['x'].append(object_position)
                        ts[name]['t'].append(t)

                        if not object_histogram is None:
                            ts[name+'_hist']['x'].append(object_histogram)
                            ts[name+'_hist']['t'].append(t)

                        ts[name+'_color']['x'].append(object_color)
                        ts[name+'_color']['t'].append(t)

                        ts[name+'_volume']['x'].append(object_volume)
                        ts[name+'_volume']['t'].append(t)

            elif topic == eef_topic:
                gripper_position = get_gripper_position(msg) 
                ts['gripper_position']['x'].append(gripper_position)
                ts['gripper_position']['t'].append(t)

            elif topic == gripper_state_topic:
                gripper_state = get_gripper_state(msg)
                ts['gripper_state']['x'].append(gripper_state)
                ts['gripper_state']['t'].append(t)

            else:
                force = get_force(msg)
                torque = get_torque(msg)

                ts['force']['x'].append(force)
                ts['force']['t'].append(t)

                ts['torque']['x'].append(torque)
                ts['torque']['t'].append(t)                

    make_np(ts['table'])
    make_np(ts['gripper_position'])
    make_np(ts['gripper_state'])
    make_np(ts['force'])
    make_np(ts['torque'])
    for name in object_filters:
        make_np(ts[name])
        make_np(ts[name+'_hist'])
        make_np(ts[name+'_color'])
        make_np(ts[name+'_volume'])

    # zero time
    ts['table']['t'] -= start_t
    ts['gripper_position']['t'] -= start_t
    ts['gripper_state']['t'] -= start_t
    ts['force']['t'] -= start_t
    ts['torque']['t'] -= start_t
    for name in object_filters:
        ts[name]['t'] -= start_t
        ts[name+'_hist']['t'] -= start_t
        ts[name+'_color']['t'] -= start_t
        ts[name+'_volume']['t'] -= start_t

    return ts

# Input: name of bag file
#        dictionary of object filters
#        resampling rate
# Returns: name of each feature element
#          resampled and smoothed feature vector data
def extract_feature_time_series(bag_file, object_filters, hz=10):
    ts = extract_state_time_series(bag_file, object_filters)
    
    min_t = max([ts[k]['t'][0] for k in ts])
    max_t = min([ts[k]['t'][-1] for k in ts])

    #print bag_file, min_t, max_t
    #print {k: ts[k]['t'][0] for k in ts}
    #print {k: ts[k]['t'][-1] for k in ts}

    for k in ts:
        x = ts[k]['x']
        t = ts[k]['t']
        x = smooth(x)
        x, t = resample(x, t, min_t, max_t, hz)
        #x, t = resample(ts[k]['x'], ts[k]['t'], min_t, max_t, hz)
        #x = smooth(x)

        ts[k]['x'] = np.array(x)
        ts[k]['t'] = np.array(t)

    fv_names, fv_ts = get_feature_vectors({k: ts[k]['x'] for k in ts})
    av_names, av_ts = get_action_vectors(ts)

    return fv_names, fv_ts, av_names, av_ts, np.array(t)

# Input: name of bag file
# Returns: np array of keyframe times
def get_keyframe_times(bag_file, add_gripper_kfs=True):
    kf_topic = '/kf_tracker/state'
    gripper_state_topic = '/vector/right_gripper/stat'

    with rosbag.Bag(bag_file) as bag:
        bag_info = yaml.load(bag._get_yaml_info())
        start_t = bag_info['start']
        end_t = bag_info['end']

        kf = []
        kf_state = 0

        gripper_state = []
        gripper_t = []

        for topic, msg, t in bag.read_messages(topics=[kf_topic, gripper_state_topic]):
            t = t.to_sec()

            if topic == kf_topic:
                if msg.data != kf_state:
                    kf.append(t)
                    kf_state = msg.data
            else:
                gripper_state.append(get_gripper_state(msg))
                gripper_t.append(t)

    if add_gripper_kfs:
        gripper_state, gripper_t = resample_interp(gripper_state,
                                                   gripper_t,
                                                   np.min(gripper_t),
                                                   np.max(gripper_t),
                                                   10)
        gripper_state = smooth(gripper_state)

        d_gripper_state = (gripper_state[1:] - gripper_state[:-1]) / (gripper_t[1:] - gripper_t[:-1])
        gripper_change = map(lambda x: x > 0.0025, np.abs(d_gripper_state))
        gripper_change = map(lambda x: x[0] != x[1], zip(gripper_change[:-1], gripper_change[1:]))
        gripper_change_idxs = [i for i, x in enumerate(gripper_change) if x]

        gripper_kfs = gripper_t[gripper_change_idxs]
        kf.extend(list(gripper_kfs))
        kf = np.sort(kf)
 
        ''' 
        plt.plot(gripper_t, gripper_state)
        plt.plot(gripper_t[:-1], d_gripper_state)
        plt.plot(gripper_t[:-1], np.abs(d_gripper_state))
        plt.plot(gripper_t[:-1], map(lambda x: 0.1 if x > 0.0025 else 0, np.abs(d_gripper_state)))
        plt.plot(gripper_t[:-2], map(lambda x: 0.1 if x else 0, gripper_change))
        plt.show()
        '''

    kf = np.array(kf)
    kf = kf - start_t
    kf = np.array([0] + list(kf) + [end_t - start_t])

    return kf

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

def add_demo(bag_file, dataset, pid, task, demo_id, object_filters, steps):
    if not pid in dataset:
        dataset[pid] = {}

    if not task in dataset[pid]:
        dataset[pid][task] = {}

    kf = get_keyframe_times(bag_file)
    state_names, state, action_names, action, t = extract_feature_time_series(bag_file, object_filters) 
    dataset[pid][task][demo_id] = {'state_names': state_names, 
                                   'state': state,
                                   'action_names': action_names,
                                   'action': action,
                                   't': t,
                                   'kf': kf}

    dataset[pid][task][demo_id]['steps'] = steps

def main():
    parser = argparse.ArgumentParser(description='Compile demonstration data')
    parser.add_argument('--src', metavar='DIR', required=True, help='Path to directory containing bag files')
    parser.add_argument('--target', metavar='DIR', required=True, help='Path to directory for saved dataset file')
    parser.add_argument('--steps', metavar='CSV', required=True, help='Name of csv steps file')
    parser.add_argument('--filename', metavar='PKL', required=True, help='Name of pkl dataset file')
    parser.add_argument('--filters', metavar='YAML', required=True, help='Path to yaml task filter parameter file')
    parser.add_argument('--tasks', metavar='TASK', nargs='+', default=[], required=False, help='Task to compile')
    parser.add_argument('--resample', metavar='RSMP', choices=['clamp', 'interp'], default='clamp', required=False, help='Resample type')
    
    args = parser.parse_args()
    src = args.src
    tar = args.target
    steps = args.steps
    filename = join(expanduser('~'), tar, args.filename)
    filters = args.filters
    tasks = args.tasks
    rsmp_type = args.resample

    global resample
    if rsmp_type == 'clamp':
        resample = resample_clamp
    else:
        resample = resample_interp

    check_compatibility(src, steps, tasks)
    steps = get_steps(steps)
    
    ensure_dir(filename)

    demos_cnt = {}
    for pid in steps:
        for task in steps[pid]:
            if not pid in demos_cnt:
                demos_cnt[pid] = {}
            
            if not task in demos_cnt[pid]:
                demos_cnt[pid][task] = 0

    task_filters = load_tasks(filters)
    bag_files = sort_by_timestamp(get_files_recursive(src, dirs_get=tasks, type='bag'))
    n = len(bag_files)

    dataset = {}
    for i, bag_file in enumerate(bag_files):
        print 'Processing ' + bag_file + ' (' + str(i+1) + '/' + str(n) + ')...'

        pid = get_task_name(bag_file)
        task = get_skill_name(bag_file)

        demo_idx = demos_cnt[pid][task]
        demo_num = steps[pid][task].keys()[demo_idx]
        demo_id = 'd'+str(demo_num)+'_'+get_timestamp(bag_file)

        add_demo(bag_file, dataset, pid, task, demo_id, task_filters[task], steps[pid][task][demo_num])

        demos_cnt[pid][task] += 1

    pickle.dump(dataset, open(filename, 'w'))

if __name__ == '__main__':
    main()

