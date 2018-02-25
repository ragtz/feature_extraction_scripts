#!/usr/bin/env python

from hlpr_perception_msgs.msg import ExtractedFeaturesArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from object_filters import *
import rospy
import sys

class ObjectsViz:
    def __init__(self, features_topic, obj_filters):
        self.features_topic = features_topic
        self.obj_filters = obj_filters
        self.viz_pub = None

    def get_cube_marker(self, name, obj):
        marker = Marker()

        marker.header = obj.header
        marker.ns = name
        marker.id = 0
        marker.type = Marker.CUBE

        marker.pose.position = Point(obj.obb.bb_center.x,
                                     obj.obb.bb_center.y,
                                     obj.obb.bb_center.z)
        marker.pose.orientation = obj.obb.bb_rot_quat

        marker.scale = Vector3(obj.obb.bb_dims.x,
                               obj.obb.bb_dims.y,
                               obj.obb.bb_dims.z)

        marker.color = obj.basicInfo.rgba_color

        return marker

    def get_text_marker(self, name, obj):
        marker = Marker()

        marker.header = obj.header
        marker.ns = name + '_text'
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING

        marker.pose = Point(obj.obb.bb_center.x, 
                            obj.obb.bb_center.y,
                            obj.obb.bb_center.z)

        marker.scale.z = 0.05
        marker.color = ColorRGBA(1,1,1,1)

        marker.text = name

        return marker

    def publish_objects(self, msg):
        markers = []

        for name, obj_filter in self.obj_filters.iteritems():
            obj_idx = obj_filter(msg.objects)
            if obj_idx:
                obj = msg.objects[obj_idx[0]]
                markers.append(get_cube_marker(name, obj))
                markers.append(get_text_marker(name, obj))
                 
        self.viz_pub.publish(MarkerArray(markers))

    def run(self):
        rospy.init_node('objects_viz')
        
        rospy.Subscriber(self.features_topic, ExtractedFeaturesArray, self.publish_objects)
        self.viz_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        
        rospy.spin()

def main():
    task_yaml = sys.argv[1]
    obj_filters = object_filters_from_yaml(task_yaml)

    objects_viz = ObjectsViz('/beliefs/features', obj_filters)
    objects_viz.run()

if __name__ == '__main__':
    main()

