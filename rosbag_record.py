#!/usr/bin/env python

from os.path import dirname, join, expanduser, exists
from os import makedirs, system
import subprocess

def terminate_ros_node(s):
    list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
    list_output = list_cmd.stdout.read()
    retcode = list_cmd.wait()
    assert retcode == 0, "List command returned %d" % retcode
    for str in list_output.split("\n"):
        if (str.startswith(s)):
            system("rosnode kill " + str)

def ensure_dir(f):
    d = dirname(f)
    if not exists(d):
        makedirs(d)

def start_record(path, filename, topics):
    datapath = join(expanduser("~"), path, filename)
    
    # Check if directory exists
    ensure_dir(datapath)
   
    # Setup the command for rosbag
    # We don't use the compression flag (-j) to avoid slow downs
    rosbag_cmd = " ".join(("rosbag record -O", datapath, topics))

    # Start the command through the system
    rosbag_proc = subprocess.Popen([rosbag_cmd], shell=True)
    
    return rosbag_proc

def stop_record(rosbag_proc):
    # Send command to the process to end (same as ctrl+C)
    rosbag_proc.send_signal(subprocess.signal.SIGINT)

    # Kill all extra rosbag "record" nodes
    terminate_ros_node("/record")

