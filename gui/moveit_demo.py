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
from ljcmp.utils.time_parameterization import time_parameterize
from ljcmp.planning.sample_region import RegionSampler, LatentRegionSampler

from ljcmp.utils.rospy_utils import interactive_marker_control_4dof
from ljcmp.utils.generate_environment import generate_environment
from ljcmp.planning.constrained_bi_rrt_latent_jump import ConstrainedLatentBiRRT


import rospy
from visualization_msgs.msg import InteractiveMarkerUpdate, InteractiveMarkerFeedback, InteractiveMarker, InteractiveMarkerControl
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from scipy.spatial.transform import Rotation as R

from moveit_msgs.msg import RobotState, DisplayRobotState, DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

model_info = yaml.load(open('model/panda_dual_orientation/model_info.yaml', 'r'), Loader=yaml.FullLoader)
constraint, model_info, c, update_scene_from_yaml, set_constraint, q_init = generate_environment('panda_dual_orientation')
constraint_model, validity_model = load_model('panda_dual_orientation', model_info, True)

constraint.set_early_stopping(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

constraint_model = constraint_model.to(device)
validity_model = validity_model.to(device)

robot_state_pub = rospy.Publisher('/display_ik_robot_state', DisplayRobotState, queue_size=1)
display_trajectory_pub = rospy.Publisher('/display_planned_path', DisplayTrajectory, queue_size=1)

last_trajectory = None
last_q = np.zeros_like(q_init)
def to_robot_state(joint_values):
    display_robot_state = DisplayRobotState()
    robot_state = RobotState()
    
    robot_state.joint_state.name = ['panda_2_joint1', 'panda_2_joint2', 'panda_2_joint3', 'panda_2_joint4', 'panda_2_joint5', 'panda_2_joint6', 'panda_2_joint7',
                                    'panda_1_joint1', 'panda_1_joint2', 'panda_1_joint3', 'panda_1_joint4', 'panda_1_joint5', 'panda_1_joint6', 'panda_1_joint7']
    robot_state.joint_state.position = joint_values.tolist()
    # robot_state.header.frame_id = 'base'
    # robot_state.header.stamp = rospy.Time.now()
    display_robot_state.state = robot_state
    return display_robot_state

def to_trajectory_msg(duration, q, qdot, qddot, t):
    display_trajectory = DisplayTrajectory()
    display_trajectory.trajectory.append(RobotTrajectory())
    display_trajectory.trajectory[0].joint_trajectory.header.frame_id = 'base'
    display_trajectory.trajectory[0].joint_trajectory.header.stamp = rospy.Time.now()

    display_trajectory.trajectory[0].joint_trajectory.joint_names = ['panda_2_joint1', 'panda_2_joint2', 'panda_2_joint3', 'panda_2_joint4', 'panda_2_joint5', 'panda_2_joint6', 'panda_2_joint7',
                                    'panda_1_joint1', 'panda_1_joint2', 'panda_1_joint3', 'panda_1_joint4', 'panda_1_joint5', 'panda_1_joint6', 'panda_1_joint7']
    
    for q_i, qdot_i, qddot_i, t_i in zip(q, qdot, qddot, t):
        display_trajectory.trajectory[0].joint_trajectory.points.append(JointTrajectoryPoint(positions=q_i.tolist(), time_from_start=rospy.Duration(t_i)))

    return display_trajectory


lrs = LatentRegionSampler(constraint_model, constraint, validity_model, model_sample_count=512, new_config_threshold=0.5)
# lrs.debug = True

def callback(data):
    global q_init
    print(colored('start solving IK', 'green'))
    pose = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])
    # print(colored('target_pose', 'green'), pose)
    global lrs
    lrs.set_target_pose(pose)
    
    q = lrs.sample(0.25, q0=q_init.copy()) # timeout 0.5s
    if q is None:
        print(colored('no solution', 'red'))
    else:

        # if constraint.planning_scene.is_valid(q) is False:
        #     print('what?')
        #     constraint.planning_scene.update_joints(q)
        #     constraint.planning_scene.print_current_collision_infos()
        #     return
        # print(q)
        robot_state_pub.publish(to_robot_state(q))
        planner = ConstrainedLatentBiRRT(constraint_model, validity_model, constraint, latent_jump=True)
                                                    # goal_region_fn=lrs.sample)
        planner.set_start(q_init)
        planner.set_goal(q)

        planner.max_distance = model_info['planning']['max_distance_q'] / model_info['planning']['alpha']
        planner.max_distance_q = model_info['planning']['max_distance_q']
        planner.off_manifold_threshold = model_info['planning']['off_manifold_threshold']
        planner.p_q_plan = model_info['planning']['p_q_plan']
        print(colored('Start Solving Path', 'green'))
        r, z_path, q_path, path_ref = planner.solve(max_time=1.5)
        if r:
            duration, qs_sample, qds_sample, qdds_sample, ts_sample = time_parameterize(q_path, model_info, hz=15)
            display_trajectory = to_trajectory_msg(duration, qs_sample, qds_sample, qdds_sample, ts_sample)
            display_trajectory_pub.publish(display_trajectory)
            global last_q, last_trajectory
            last_q = q_path[-1]
            last_trajectory = qs_sample
            print(colored('success', 'green'))
            # print(colored('z_path', 'green'), z_path)
            # print(colored('q_path', 'green'), q_path)
        else:
            print(colored('fail to plan', 'red'))
        # planner.set_goal(q)

        # planner.set_go(lrs)


    # print(data)


server = InteractiveMarkerServer("test",q_size=1)

marker_pos = np.array([0.43, 0.47, 0.83])
marker_quat = np.array([0,0,0, 1.0])

update = InteractiveMarkerUpdate()
update.type = InteractiveMarkerUpdate.KEEP_ALIVE

marker = InteractiveMarker()
marker.controls = interactive_marker_control_4dof()
marker.header.frame_id = 'base'
marker.name = 'test'
marker.pose.position.x = marker_pos[0]
marker.pose.position.y = marker_pos[1]
marker.pose.position.z = marker_pos[2]
marker.pose.orientation.x = marker_quat[0]
marker.pose.orientation.y = marker_quat[1]
marker.pose.orientation.z = marker_quat[2]
marker.pose.orientation.w = marker_quat[3]
marker.scale = 0.3

update.markers.append(marker)
update.server_id = 'test'

server.insert(marker, callback)
server.applyChanges()

    # interactive_marker_pub.publish(update)

    # time.sleep(1.0)
constraint.planning_scene.add_box('obj', [0.5, 0.06, 0.7], [0.55, 0.0, 0.67], [0,0,0,1])
# constraint.planning_scene.add_box('obj2', [0.1, 0.1, 0.1], [0.3, 0.5, 0.9], [0,0,0,1])
# constraint.planning_scene.add_box('obj3', [0.1, 0.1, 0.1], [0.5, -0.5, 0.7], [0,0,0,1])
# constraint.planning_scene.add_box('obj4', [0.1, 0.1, 0.1], [0.7, 0.8, 1.2], [0,0,0,1])
# constraint.planning_scene.add_box('obj5', [0.1, 0.1, 0.1], [0.3, -0.8, 1.2], [0,0,0,1])
# constraint.planning_scene.add_box('obj2', [0.8, 0.8, 0.07], [0.55, 0.0, 1.5], [0,0,0,1])
# [ 1.6594 -1.4994 -1.617  -1.2433 -2.1471  1.7362  0.6078 -0.2573  1.0647 -0.937  -1.0909  1.1271  1.2139 -0.2783]
q_init = np.array([ 1.6594, -1.4994, -1.617,  -1.2433, -2.1471,  1.7362,  0.6078, -0.2573,  1.0647, -0.937,  -1.0909,  1.1271,  1.2139, -0.2783])
last_q = q_init.copy()
constraint.planning_scene.update_joints(q_init)
constraint.planning_scene.display()

np.set_printoptions(precision=4, suppress=True)
while not rospy.is_shutdown():
    r = input('press enter to update target')
    for q in last_trajectory:
        # print(q)
        constraint.planning_scene.update_joints(q)
        constraint.planning_scene.display()
        time.sleep(0.05)
    q_init = last_q.copy()
    print(colored('q_init', 'green'), q_init)
    # constraint.planning_scene.display()
    # time.sleep(0.5)
    # pass

rospy.spin()    