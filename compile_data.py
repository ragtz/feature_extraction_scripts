#!/usr/bin/env python

from geometry_msgs.msg import PointStamped
from os.path import join, expanduser, splitext, basename
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from tf.transformations import compose_matrix
from tf import TransformerROS
from yaml_include_loader.loader import *
from object_filters import *
from fetch_files import *
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import itertools
import argparse
import pickle
import rosbag
import re

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
# Returns: rgb np array
def get_object_color(obj):
    rgba = obj.basicInfo.rgba_color
    return np.array([rgba.r, rgba.g, rgba.b])

# Input: Pose msg
# Returns: position np array
def get_gripper_position(msg):
    return np.array([msg.position.x, msg.position.y, msg.position.z])

# Input: GripperStat msg
# Returns: open gripper state
def get_gripper_state(msg):
    return msg.position

def resample(x, t, min_t, max_t, hz):
    f = interp1d(t, x, kind='linear', axis=0)
    t_rs = np.arange(min_t, max_t, 1.0/hz)
    x_rs = f(t_rs)
    return x_rs, t_rs

def smooth(x):
    b, a = butter(2, 0.125)
    return filtfilt(b, a, x, axis=0, method='gust')

def is_hist(name):
    return '_hist' in name

def is_color(name):
    return '_color' in name

def is_volume(name):
    return '_volume' in name

def is_obj(name):
    return not is_hist(name) and not is_color(name) and not is_volume(name)

# Input: state dictionary of form {'name': np.array([0,0,0]), ...}
#        list of objects to get histogram features
#        list of objects to get color features
# Returns: feature names, features tuple
def get_feature_vector(state, obj_hists=['pitcher'], obj_colors=['lamp'], obj_volume=['drawer']):
    table = state['table']
    gripper_position = state['gripper_position']
    gripper_state = state['gripper_state']
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

    start_t = float('inf')
    with rosbag.Bag(bag_file) as bag:
        for _, _, t in bag.read_messages():
            t = t.to_sec()
            if t < start_t:
                start_t = t

    with rosbag.Bag(bag_file) as bag:
        ts = {'table': {'x': [], 't': []},
              'gripper_position': {'x': [], 't': []},
              'gripper_state': {'x': [], 't': []}}
        for name in object_filters:
            ts[name] = {'x': [], 't': []}
            ts[name+'_hist'] = {'x': [], 't': []}
            ts[name+'_color'] = {'x': [], 't': []}
            ts[name+'_volume'] = {'x': [], 't': []}

        for topic, msg, t in bag.read_messages(topics=[features_topic, eef_topic, gripper_state_topic]):
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

            else:
                gripper_state = get_gripper_state(msg)
                ts['gripper_state']['x'].append(gripper_state)
                ts['gripper_state']['t'].append(t)

    make_np(ts['table'])
    make_np(ts['gripper_position'])
    make_np(ts['gripper_state'])
    for name in object_filters:
        make_np(ts[name])
        make_np(ts[name+'_hist'])
        make_np(ts[name+'_color'])
        make_np(ts[name+'_volume'])

    # zero time
    ts['table']['t'] -= start_t
    ts['gripper_position']['t'] -= start_t
    ts['gripper_state']['t'] -= start_t
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

def get_timestamp(f, format='%Y-%m-%dT%H%M%S'):
    return re.split('_', splitext(basename(f))[0])[-1]

def add_demo(bag_file, dataset, pid, task, object_filters):
    if not pid in dataset:
        dataset[pid] = {}

    if not task in dataset[pid]:
        dataset[pid][task] = {}

    demo_num = len(dataset[pid][task])
    state_names, state, action_names, action, t = extract_feature_time_series(bag_file, object_filters) 
    dataset[pid][task]['d'+str(demo_num)+'_'+get_timestamp(bag_file)] = {'state_names': state_names, 
                                                                         'state': state,
                                                                         'action_names': action_names,
                                                                         'action': action,
                                                                         't': t}

def main():
    parser = argparse.ArgumentParser(description='Compile demonstration data')
    parser.add_argument('--src', metavar='DIR', required=True, help='Path to directory containing bag files')
    parser.add_argument('--target', metavar='DIR', required=True, help='Path to directory for saved bag files')
    parser.add_argument('--filename', metavar='PKL', required=True, help='Name of pkl dataset file')
    parser.add_argument('--filters', metavar='YAML', required=True, help='Path to yaml task filter parameter file')
    parser.add_argument('--tasks', metavar='TASK', nargs='+', default=[], required=False, help='Task to compile')
    
    args = parser.parse_args()
    src = args.src
    tar = args.target
    filename = join(expanduser('~'), tar, args.filename)
    filters = args.filters
    tasks = args.tasks

    ensure_dir(filename)

    task_filters = load_tasks(filters)
    bag_files = sort_by_timestamp(get_files_recursive(src, dirs_get=tasks, type='bag'))
    n = len(bag_files)

    dataset = {}
    for i, bag_file in enumerate(bag_files):
        print 'Processing bag file ' + str(i+1) + ' of ' + str(n) + '...'

        pid = get_task_name(bag_file)
        task = get_skill_name(bag_file)
        add_demo(bag_file, dataset, pid, task, task_filters[task])

    pickle.dump(dataset, open(filename, 'w'))

if __name__ == '__main__':
    main()

