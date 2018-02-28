#!/usr/env/bin python

from sensor_msgs.msg import JointState
import rospy

tilt_idx = 15
min_tilt = 0.49
max_tilt = 0.51

def filter_tilt(msg, pub):
    tilt = msg.position[tilt_idx]
    if tilt > min_tilt and tilt < max_tilt:
        pub.publish(msg)

def main():
    rospy.init_node('tilt_filter')
    pub = rospy.Publisher('joint_states', JointState, queue_size=10)
    rospy.Subscriber('joint_states_pre_filtered', JointState, filter_tilt, pub)
    rospy.spin()

if __name__ == '__main__':
    main()

