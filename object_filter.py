#!/usr/bin/env python

import rospy
import ctypes
import struct
import hsv_utils

from hlpr_perception_msgs.msg import SegClusters
from copy import deepcopy
import sensor_msgs.point_cloud2 as pc2
import numpy as np

pitcher_params = {"h_range": (0.53, 0.64),
                  "s_range": (0.75, 1.0),
                  "v_range": None}

mug_params = {"h_range": (0.085, 0.2),
              "s_range": (0.8, 1.0),
              "v_range": None}
'''
mug_params = {"h_range": (0.095, 0.2),
              "s_range": (0.73, 1.0),
              "v_range": None}
'''

def read_rgb(p):
    rgb = p[3]

    # cast float32 to int so that bitwise operations are possible
    s = struct.pack('>f', rgb)
    i = struct.unpack('>l', s)[0]

    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)

    return (r, g, b)

def get_avg_rgb(pc):
    r_sum = 0
    g_sum = 0
    b_sum = 0
    n = 0

    points = pc2.read_points(pc, skip_nans=True)
    for point in points:
        r, g, b = read_rgb(point)
        r_sum += r
        g_sum += g
        b_sum += b
        n += 1

    return ((r_sum/n)/255.0, (g_sum/n)/255.0, (b_sum/n)/255.0)

def get_avg_hsv(pc):
    r, g, b = get_avg_rgb(pc)
    return hsv_utils.rgb_to_hsv(r, g, b) 

def hsv_filter(h_range, s_range, v_range):
    def f(msg):
        keep_idxs = []
        for idx, cluster in enumerate(msg.clusters):
            h, s, v = get_avg_hsv(cluster)
            #print h, s, v
            if hsv_utils.in_hsv_range(h, s, v, h_range, s_range, v_range):
                keep_idxs.append(idx)

        f_msg = SegClusters()
        f_msg.header = deepcopy(msg.header)
        f_msg.header.frame_id = 'kinect_ir_optical_frame'
        f_msg.clusters = np.array(msg.clusters)[keep_idxs]
        f_msg.normals = np.array(msg.normals)[keep_idxs]
        f_msg.plane = msg.plane
        f_msg.cluster_ids = np.array(msg.cluster_ids)[keep_idxs]

        return f_msg
    return f

def largest_filter(msg):
    if len(msg.clusters) > 0:
        det_cov = []
        for cluster in msg.clusters:
            points = pc2.read_points(cluster, skip_nans=True)
            pts = []
            
            for idx, point in enumerate(points):
                if idx%10 == 0:
                    pts.append([point[0], point[1], point[2]])

            if len(pts) > 1:
                det_cov.append(np.linalg.det(np.cov(np.array(pts).T)))
            else:
                det_cov.append(-float("inf"))
       
        l_idx = np.argmax(det_cov)
     
        f_msg = SegClusters()
        f_msg.header = deepcopy(msg.header)
        f_msg.clusters = [msg.clusters[l_idx]]
        f_msg.normals = [msg.normals[l_idx]]
        f_msg.plane = msg.plane
        f_msg.cluster_ids = [msg.cluster_ids[l_idx]]

        return f_msg
    else:
        return msg

def object_filter(params):
    h_range = params["h_range"]
    s_range = params["s_range"]
    v_range = params["v_range"]

    def f(msg):
        return largest_filter(hsv_filter(h_range, s_range, v_range)(msg))
        #return hsv_filter(h_range, s_range, v_range)(msg)
 
    return f

def pitcher_filter(msg):
    return object_filter(pitcher_params)(msg)

def mug_filter(msg):
    return object_filter(mug_params)(msg)

def merge_clusters(msgs):
    m_msg = SegClusters()
    m_msg.header = deepcopy(msgs[0].header)
    m_msg.clusters = []
    m_msg.normals = []
    m_msg.plane = deepcopy(msgs[0].plane)
    m_msg.cluster_ids = []

    for msg in msgs:
        m_msg.clusters.extend(msg.clusters)
        m_msg.normals.extend(msg.normals)
        m_msg.cluster_ids.extend(msg.cluster_ids)

    return m_msg

class ObjectsFilter:
    def __init__(self, pre_filter_topic, post_filter_topic, obj_filters):
        self.pre_filter_topic = pre_filter_topic
        self.post_filter_topic = post_filter_topic
        self.obj_filters = obj_filters
        self.pub = None

    def filter(self, msg):
        obj_clusters = []

        for obj_filter in self.obj_filters:
            obj_clusters.append(obj_filter(msg))
        
        self.pub.publish(merge_clusters(obj_clusters))

    def run(self):
        rospy.init_node('objects_filter')
        
        rospy.Subscriber(self.pre_filter_topic, SegClusters, self.filter)
        self.pub = rospy.Publisher(self.post_filter_topic, SegClusters, queue_size=10)
        
        rospy.spin()
        
class ArmFilter:
    def __init__(self, pre_filter_topic, post_filter_topic, h_range, s_range, v_range):
        self.pre_filter_topic = pre_filter_topic
        self.post_filter_topic = post_filter_topic
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range
        self.pub = None

    def readRGB(self, p):
        rgb = p[3]

        # cast float32 to int so that bitwise operations are possible
        s = struct.pack('>f', rgb)
        i = struct.unpack('>l', s)[0]

        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)

        return (r, g, b)

    def getAvgRGB(self, pc):
        r_sum = 0
        g_sum = 0
        b_sum = 0
        n = 0

        points = pc2.read_points(pc, skip_nans=True)
        for point in points:
            r, g, b = self.readRGB(point)
            r_sum += r
            g_sum += g
            b_sum += b
            n += 1

        return ((r_sum/n)/255.0, (g_sum/n)/255.0, (b_sum/n)/255.0)

    def isArmCluster(self, pc):
        r, g, b = self.getAvgRGB(pc)
        h, s, v = hsv_utils.rgb_to_hsv(r, g, b)
        print "========> HSV:", h, s, v 
        return hsv_utils.in_hsv_range(h, s, v, self.h_range, self.s_range, self.v_range)

    def filter(self, msg):
        keep_idxs = []
        for idx, cluster in enumerate(msg.clusters):
            if self.isArmCluster(cluster):
                keep_idxs.append(idx)

        print "--------------------"
        print "Removed", len(msg.clusters)-len(keep_idxs), "clusters..."
        print "--------------------"

        filtered_msg = SegClusters()
        filtered_msg.clusters = np.array(msg.clusters)[keep_idxs]
        filtered_msg.normals = np.array(msg.normals)[keep_idxs]
        filtered_msg.plane = msg.plane
        filtered_msg.cluster_ids = np.array(msg.cluster_ids)[keep_idxs]

        self.pub.publish(filtered_msg)

    def run(self):
        rospy.init_node('arm_filter')
        
        rospy.Subscriber(self.pre_filter_topic, SegClusters, self.filter)
        self.pub = rospy.Publisher(self.post_filter_topic, SegClusters)
        
        rospy.spin()

if __name__ == "__main__":
    '''    
    pre_filter_topic = rospy.get_param('/arm_filter/pre_filter')
    post_filter_topic = rospy.get_param('/arm_filter/post_filter')

    min_h = rospy.get_param('/arm_filter/min_h')
    max_h = rospy.get_param('/arm_filter/max_h')

    min_s = rospy.get_param('/arm_filter/min_s')
    max_s = rospy.get_param('/arm_filter/max_s')

    min_v = rospy.get_param('/arm_filter/min_v')
    max_v = rospy.get_param('/arm_filter/max_v')

    h_range = (min_h, max_h) if min_h >= 0 and max_h >= 0 else None
    s_range = (min_s, max_s) if min_s >= 0 and max_s >= 0 else None
    v_range = (min_v, max_v) if min_v >= 0 and max_v >= 0 else None

    print "--------------------"
    print h_range
    print s_range
    print v_range
    print "--------------------"

    arm_filter = ArmFilter(pre_filter_topic, post_filter_topic, h_range, s_range, v_range)
    arm_filter.run()
    '''
    
    pre_filter_topic = rospy.get_param('/objs_filter/pre_filter')
    post_filter_topic = rospy.get_param('/objs_filter/post_filter')

    objs_filter = ObjectsFilter(pre_filter_topic, post_filter_topic, [pitcher_filter, mug_filter])
    #objs_filter = ObjectsFilter(pre_filter_topic, post_filter_topic, [pitcher_filter])
    #objs_filter = ObjectsFilter(pre_filter_topic, post_filter_topic, [mug_filter]) 
    objs_filter.run()
    

