#!/usr/bin/env python

from __future__ import print_function

from std_srvs.srv import Trigger
import rospy

def handle_add_two_ints(req):
    print("Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b)))
    return Trigger(req.a + req.b)

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('add_two_ints', Trigger, handle_add_two_ints)
    print("Ready to add two ints.")
    while not rospy.is_shutdown():
        rospy.sleep(1)

if __name__ == "__main__":
    add_two_ints_server()