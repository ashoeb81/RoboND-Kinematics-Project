#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
import numpy as np
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *


def get_rot_x(roll):
    """Return rotation matrix about x-axis

    :param roll: Desired angle of rotation in radians.
    :return: Numpy array.
    """
    rotm = np.array([[1., 0., 0.],
                     [0., cos(roll), -sin(roll)],
                     [0., sin(roll), cos(roll)]], dtype=float)
    return rotm


def get_rot_y(pitch):
    """Return rotation matrix about y-axis.

    :param pitch: Desired angle of rotation in radians.
    :return: Numpy array
    """
    rotm = np.array([[cos(pitch), 0., sin(pitch)],
                     [0., 1., 0.],
                     [-sin(pitch), 0., cos(pitch)]], dtype=float)
    return rotm


def get_rot_z(yaw):
    """Return rotation matrix about z-axis

    :param yaw: Desired angle of rotation in radians.
    :return: Numpy array.
    """
    rotm = np.array([[cos(yaw), -sin(yaw), 0.],
                     [sin(yaw), cos(yaw), 0.],
                     [0., 0., 1.]], dtype=float)
    return rotm


def get_transform_matrix(alpha,  a, q, d):
    """Return homogenuous transformation from frame i-1 to frame i using DH parameters.

    :param alpha: Angle between Z(i-1) and Z(i) axes about X(i-1) axis.
    :param a: Distance between Z(i-1) and Z(i) axes measured about X(i) axis.
    :param q: Angle between X(i-1) and X(i) axes about Z(i) axis.
    :param d: Distance between X(i-1) and X(i) axis measured about Z(i) axis.
    :return: Sympy Matirx.
    """
    T = Matrix([[       cos(q),             -sin(q),            0,                 a],
                [sin(q)*cos(alpha),  cos(q)*cos(alpha), -sin(alpha),  -sin(alpha)*d],
                [sin(q)*sin(alpha),  cos(q)*sin(alpha),  cos(alpha),   cos(alpha)*d],
                [0, 0, 0, 1]])
    return T

def get_wrist_coords(eef_coords, roll, pitch, yaw):
    """Return Wrist-Center coordinates in the world coordinate frame.

    :param eef_coords: End-effector (gripper) coordinates.
    :param roll: End-effector roll angle (in world coordinate frame).
    :param pitch: End-effector pitch angle (in world coordinate frame).
    :param yaw: End-effector yaw angle (in world coordinate frame).
    :return: Numpy 1D array with x,y,z coordinates of wrist center.
    """
    # Get end-effector rotation matrix Rz*Ry*Rx
    rotm = np.matmul(np.matmul(get_rot_z(yaw), get_rot_y(pitch)), get_rot_x(roll))
    # Get wrist-center coordinates win world coordinate frame.
    w_c = eef_coords - (0.193 + 0.11) * np.dot(rotm, np.array([[1.], [0.], [0.]]))
    return w_c.ravel()


def kuka_inverse_kinematics(eef_coords, roll, pitch, yaw, R0_3, q1, q2, q3):
    """Returns Kuka arm joint angles given end-effector position and pose.

    :param eef_coords: End-effector (gripper) coordinates.
    :param roll: End-effector roll angle (in world coordinate frame).
    :param pitch: End-effector pitch angle (in world coordinate frame).
    :param yaw: End-effector yaw angle (in world coordinate frame).
    :param R0_3: Rotation matrix from base frame to joint 3 frame.
    :return: Numpy array of 6 joint angles (in radians).
    """
    # Kuka Arm Geometry
    z01 = 0.33  # Z-axis displacement from joint 0 to 1
    z12 = 0.42  # Z-axis displacement form joint 1 to 2
    z23 = 1.25  # Z-axis displacement from joint 2 to 3
    z34 = 0.054  # Z-axis displacement from joint 3 to 4

    x12 = 0.35  # X-axis displacement from joint 1 to 2
    x34 = 0.96  # X-axis displacement form joint 3 to 4
    x45 = 0.54  # X-axis displacement from joint 4 to 5
    x35 = x34 + x45  # X-axis displacement from joint 3 to 5

    # Wrist center coordinates in world frame
    wc = get_wrist_coords(eef_coords, roll, pitch, yaw)

    # Angle theta 1
    theta1 = np.arctan2(wc[1], wc[0])

    # Distance between Joints 2 and wrist center
    wc_2 = wc - np.array([x12 * np.cos(theta1), x12 * np.sin(theta1), z01 + z12])
    wc_2_norm = np.linalg.norm(wc_2)

    # Distance between Joint 3 and wrist center
    wc_3_norm = np.linalg.norm([z34, x35])

    # Angle theta 2
    psi = np.arccos((z23 ** 2 + wc_2_norm ** 2 - wc_3_norm ** 2) / (2 * z23 * wc_2_norm))
    theta2 = np.pi / 2 - psi - np.arctan2(wc_2[2], np.linalg.norm(wc_2[:2]))

    # Angle theta 3
    psi = np.arccos((z23 ** 2 + wc_3_norm ** 2 - wc_2_norm ** 2) / (2 * z23 * wc_3_norm))
    theta3 = np.pi / 2 - psi - np.arctan2(z34, x35)

    # End-Effector rotation matrix in world frame
    R0_6 = np.dot(np.dot(get_rot_z(yaw), get_rot_y(pitch)), get_rot_x(roll))

    # Compute Rotation matrix from Joint 3 to End-effector (R3_6)
    R0_3 = R0_3.evalf(subs={q1: theta1, q2: theta2, q3: theta3}).transpose()
    R0_3 = np.array(R0_3.tolist()).astype(float)
    R3_6 = np.dot(R0_3, R0_6)
    R3_6 = np.array(R3_6).astype(float)

    # Extract angle theta 4
    theta4 = np.arctan2(R3_6[2, 0], -1 * R3_6[0, 0])

    # Extract angle theta 5
    theta5 = np.arctan2(np.sqrt(R3_6[2, 0] ** 2 + R3_6[0, 0] ** 2), R3_6[1, 0])

    # Extract angle theta 6
    theta6 = np.arctan2(R3_6[1, 1], R3_6[1, 2])

    return theta1, theta2, theta3, theta4, theta5, theta6

def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        # Initialize service response
        joint_trajectory_list = []

        # Define DH param symbols
        q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
        d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
        a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
        alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')

        # Modified DH params
        s = {alpha0: 0, a0: 0, d1: 0.75,
             alpha1: -pi / 2, a1: 0.35, d2: 0, q2: q2 - pi / 2,
             alpha2: 0, a2: 1.25, d3: 0,
             alpha3: -pi / 2, a3: -0.054, d4: 1.50,
             alpha4: pi / 2, a4: 0, d5: 0,
             alpha5: -pi / 2, a5: 0, d6: 0,
             alpha6: 0, a6: 0, d7: 0.303, q7: 0}

        # Create individual transformation matrices
        T0_1 = get_transform_matrix(alpha0, a0, q1, d1)
        T0_1 = T0_1.subs(s)

        T1_2 = get_transform_matrix(alpha1, a1, q2, d2)
        T1_2 = T1_2.subs(s)

        T2_3 = get_transform_matrix(alpha2, a2, q3, d3)
        T2_3 = T2_3.subs(s)

        # Rotation matrix from world frame to joint 3 frame
        T0_3 = simplify(T0_1 * T1_2 * T2_3)
        R0_3 = T0_3[:3, :3]

        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            # Calculate joint angles using Geometric IK method
            theta1, theta2, theta3, theta4, theta5, theta6 = kuka_inverse_kinematics(
                [[px], [py], [pz]], roll, pitch, yaw, R0_3, q1, q2, q3)

            # Populate response for the IK request
            #  In the next line replace theta1,theta2...,theta6 by your joint angle variables
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
