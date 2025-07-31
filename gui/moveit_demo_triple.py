import os
import yaml
import numpy as np
import torch
import pickle
import time

import argparse

from termcolor import colored

import ljcmp
from ljcmp.utils.model_utils import load_model
from ljcmp.planning.sample_region import RegionSampler, LatentRegionSampler

from ljcmp.utils.rospy_utils import interactive_marker_control_6dof
from ljcmp.utils.generate_environment import generate_environment
from ljcmp.planning.constrained_bi_rrt_latent_jump import ConstrainedLatentBiRRT


import rospy
from visualization_msgs.msg import InteractiveMarkerUpdate, InteractiveMarkerFeedback, InteractiveMarker, InteractiveMarkerControl
from interactive_markers.interactive_marker_server import InteractiveMarkerServer

from moveit_msgs.msg import RobotState, DisplayRobotState


model_info = yaml.load(open('model/panda_triple/model_info.yaml', 'r'), Loader=yaml.FullLoader)
constraint, model_info, c, update_scene_from_yaml, set_constraint, q_init = generate_environment('panda_triple')
constraint_model, validity_model = load_model('panda_triple', model_info, True)

constraint.set_early_stopping(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

constraint_model = constraint_model.to(device)
validity_model = validity_model.to(device)

robot_state_pub = rospy.Publisher('/display_ik_robot_state', DisplayRobotState, queue_size=1)


def to_robot_state(joint_values):
    display_robot_state = DisplayRobotState()
    robot_state = RobotState()
    
    robot_state.joint_state.name = ['panda_left_joint1', 'panda_left_joint2', 'panda_left_joint3', 'panda_left_joint4', 'panda_left_joint5', 'panda_left_joint6', 'panda_left_joint7',
                                    'panda_right_joint1', 'panda_right_joint2', 'panda_right_joint3', 'panda_right_joint4', 'panda_right_joint5', 'panda_right_joint6', 'panda_right_joint7',
                                    'panda_top_joint1', 'panda_top_joint2', 'panda_top_joint3', 'panda_top_joint4', 'panda_top_joint5', 'panda_top_joint6', 'panda_top_joint7']
    robot_state.joint_state.position = joint_values.tolist()
    # robot_state.header.frame_id = 'base'
    # robot_state.header.stamp = rospy.Time.now()
    display_robot_state.state = robot_state
    return display_robot_state

lrs = LatentRegionSampler(constraint_model, constraint, validity_model)

def callback(data):
    print(colored('start solving', 'green'))
    pose = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])
    print(colored('target_pose', 'green'), pose)
    global lrs
    lrs.set_target_pose(pose)
    q = lrs.sample(1.0, q0=q_init) # timeout 0.5s
    if q is None:
        print(colored('no solution', 'red'))
    else:
        print(q)
        robot_state_pub.publish(to_robot_state(q))
        planner = ConstrainedLatentBiRRT(constraint_model, validity_model, constraint, latent_jump=True,
                                                    goal_region_fn=lrs.sample)
        planner.set_start(q_init)

        planner.max_distance = model_info['planning']['max_distance_z']
        planner.max_distance_q = model_info['planning']['max_distance_q']
        planner.off_manifold_threshold = model_info['planning']['off_manifold_threshold']
        planner.p_q_plan = model_info['planning']['p_q_plan']
        r, z_path, q_path, path_ref = planner.solve(max_time=10.0)
        if r:
            print(colored('success', 'green'))
            print(colored('z_path', 'green'), z_path)
            print(colored('q_path', 'green'), q_path)
        else:
            print(colored('fail to plan', 'red'))
        # planner.set_goal(q)

        # planner.set_go(lrs)


    # print(data)


server = InteractiveMarkerServer("test",q_size=1)


# interactive_marker_pub = rospy.Publisher('/interactive_marker_server/update', InteractiveMarkerUpdate, queue_size=1)
# interactive_marker_sub = rospy.Subscriber('/interactive_marker_server/feedback', InteractiveMarkerFeedback, callback)

chair_pos = np.array([0.69, 0.44, 1.19])
chair_quat = np.array([0.9238795, -0.3826834, 0, 0])

# while not rospy.is_shutdown():
ctrl = InteractiveMarkerControl()
ctrl.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
ctrl.orientation_mode = InteractiveMarkerControl.INHERIT
ctrl.always_visible = True
ctrl.orientation.w = 1


update = InteractiveMarkerUpdate()
update.type = InteractiveMarkerUpdate.KEEP_ALIVE

marker = InteractiveMarker()
marker.controls = interactive_marker_control_6dof()
marker.header.frame_id = 'base'
marker.name = 'test'
marker.pose.position.x = chair_pos[0]
marker.pose.position.y = chair_pos[1]
marker.pose.position.z = chair_pos[2]
marker.pose.orientation.x = chair_quat[0]
marker.pose.orientation.y = chair_quat[1]
marker.pose.orientation.z = chair_quat[2]
marker.pose.orientation.w = chair_quat[3]

update.markers.append(marker)
update.server_id = 'test'

server.insert(marker, callback)
server.applyChanges()

    # interactive_marker_pub.publish(update)

    # time.sleep(1.0)
constraint.planning_scene.display()

while not rospy.is_shutdown():
    time.sleep(0.5)
    pass

rospy.spin()    