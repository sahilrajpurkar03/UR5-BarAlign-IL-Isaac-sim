#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Image
import threading
import copy
import time
import cv2
#import csv
import os
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import numpy as np
from cv_bridge import CvBridge
from math import sin, cos, pi
import pandas as pd# pip3 install pandas pyarrow
import pyarrow as pa
import pyarrow.parquet as pq

bridge = CvBridge()

record_data = False
tool_pose_xy = [0.0, 0.0] # tool(end effector) pose
torus_pose_xyw = [0.0, 0.0, 0.0]
vid_H = 360
vid_W = 640
wrist_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
top_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
#gripper_state = 1 #1:open 0:close
action = np.array([0.0, 0.0], float)

class Get_Poses_Subscriber(Node):

    def __init__(self):
        super().__init__('get_modelstate')
        self.subscription = self.create_subscription(
            TFMessage,
            '/isaac_tf',
            self.listener_callback,
            10)
        self.subscription

        self.euler_angles = np.array([0.0, 0.0, 0.0], float)


    def listener_callback(self, data):
        global tool_pose_xy, torus_pose_xyw

        # 0:tool
        tool_pose = data.transforms[0].transform.translation
        tool_pose_xy[0] = tool_pose.y
        tool_pose_xy[1] = tool_pose.x

        # 1:torus
        torus_translation = data.transforms[1].transform.translation
        torus_rotation = data.transforms[1].transform.rotation
        torus_pose_xyw[0] = torus_translation.y
        torus_pose_xyw[1] = torus_translation.x
        self.euler_angles[:] = R.from_quat([
            torus_rotation.x,
            torus_rotation.y,
            torus_rotation.z,
            torus_rotation.w
        ]).as_euler('xyz', degrees=False)
        torus_pose_xyw[2] = self.euler_angles[2]


class Joy_Subscriber(Node):

    def __init__(self):
        super().__init__('joy_subscriber')
        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.listener_callback,
            10)
        self.subscription

        self.push_time = 0
        self.prev_push_time = 0

    def listener_callback(self, data):
        global record_data, action

        action[:] = copy.copy(data.axes[:2]) # left joy stick of PS4

        if(data.buttons[0] == 1): # X button of PS4 dualshock
            self.push_time = time.time()
            dif = self.push_time - self.prev_push_time
            if(dif > 1):
                if(record_data == False):
                    record_data = True
                    print('\033[32m'+'START RECORDING'+'\033[0m')
                elif(record_data):
                    record_data = False
                    print('\033[31m'+'END RECORDING'+'\033[0m')
            self.prev_push_time = self.push_time



class Wrist_Camera_Subscriber(Node):

    def __init__(self):
        super().__init__('wrist_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_wrist',
            self.camera_callback,
            10)
        self.subscription 

    def camera_callback(self, data):
        global wrist_camera_image
        # interpolation https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
        wrist_camera_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)


class Top_Camera_Subscriber(Node):

    def __init__(self):
        super().__init__('top_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_top',
            self.camera_callback,
            10)
        self.subscription 

    def camera_callback(self, data):
        global top_camera_image
        top_camera_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)

class Data_Recorder(Node):

    def __init__(self):
        super().__init__('Data_Recorder')
        self.Hz = 10  # bridge data frequency
        self.prev_ee_pose = np.array([0, 0, 0], float)
        self.timer = self.create_timer(1 / self.Hz, self.timer_callback)
        self.start_recording = False
        self.data_recorded = False

        #### log files for multiple runs are NOT overwritten
        base_dir = os.environ["HOME"] + "/Rahul/UR5-BarAlign-RL-Isaac-sim/ur5_simulation/src/data_collection/scripts/my_pusht/"
        self.log_dir = base_dir + "data/chunk_000/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        base_vid_dir = base_dir + 'videos/chunk_000/observation.images.'
        self.wrist_vid_dir = base_vid_dir + 'wrist/'
        if not os.path.exists(self.wrist_vid_dir):
            os.makedirs(self.wrist_vid_dir)

        self.top_vid_dir = base_vid_dir + 'top/'
        if not os.path.exists(self.top_vid_dir):
            os.makedirs(self.top_vid_dir)

        self.state_vid_dir = base_vid_dir + 'state/'
        if not os.path.exists(self.state_vid_dir):
            os.makedirs(self.state_vid_dir)

        # image of a torus shape on the table (background)
        self.initial_image = cv2.imread(os.environ['HOME'] + "/Rahul/UR5-BarAlign-RL-Isaac-sim/ur5_simulation/images/torus_top_plane.png")
        self.initial_image = cv2.rotate(self.initial_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # mask for torus region, will be drawn as circle dynamically
        self.torus_region = np.zeros((self.initial_image.shape[0], self.initial_image.shape[1]), np.uint8)

        self.pub_img = self.create_publisher(Image, '/pushT_image', 10)
        self.tool_radius = 10  # millimeters (tool circle radius)
        self.scale = 1.639344  # mm/pixel conversion scale
        self.C_W = 300  # pixel center width offset for coordinate conversion
        self.C_H = 320  # pixel center height offset for coordinate conversion

        self.radius = int(self.tool_radius / self.scale)  # radius in pixels

        self.df = pd.DataFrame(
            columns=['observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.reward', 'next.done',
                     'next.success', 'index', 'task_index'])
        self.index = 296  # starting index of data rows
        self.episode_index = 1  # starting episode index
        self.frame_index = 0
        self.time_stamp = 0.0
        self.success = False
        self.done = False
        self.column_index = 0
        self.prev_sum = 0.0

        self.wrist_camera_array = []
        self.top_camera_array = []
        self.state_image_array = []

    def timer_callback(self):
        global tool_pose_xy, torus_pose_xyw, action, wrist_camera_image, top_camera_image, record_data

        # Copy background image for this frame
        image = copy.copy(self.initial_image)
        self.torus_region[:] = 0  # reset torus region mask to blank

        # Convert torus position to pixel coordinates
        torus_x_pix = int((1000 * torus_pose_xyw[0] + self.C_W ) / self.scale)
        torus_y_pix = int((1000 * torus_pose_xyw[1] - self.C_H ) / self.scale)
        

        # Convert tool position to pixel coordinates
        tool_x_pix = int((1000 * tool_pose_xy[0] + self.C_W )/ self.scale)
        tool_y_pix = int((1000 * tool_pose_xy[1] - self.C_H) / self.scale)

        # Torus radius in pixels (adjust based on actual torus size)
        torus_radius_pix = 48

        # Draw tool circle (gray)
        cv2.circle(image, (tool_x_pix, tool_y_pix), self.radius, (100, 100, 100), thickness=cv2.FILLED)

        # Draw torus circle (red) and update torus region mask
        cv2.circle(image, (torus_x_pix, torus_y_pix), torus_radius_pix, (0, 0, 180), thickness=30)

        # For the mask (white ring with line thickness of 2 pixels)
        cv2.circle(self.torus_region, (torus_x_pix, torus_y_pix), torus_radius_pix, 255, thickness=10)


        # Create binary mask for tool circle
        tool_mask = np.zeros_like(self.torus_region)
        cv2.circle(tool_mask, (tool_x_pix, tool_y_pix), self.radius, 255, thickness=cv2.FILLED)

        # Calculate overlap area between tool and torus masks
        common_part = cv2.bitwise_and(tool_mask, self.torus_region)
        common_part_sum = cv2.countNonZero(common_part)

        torus_area = np.pi * (torus_radius_pix ** 2)
        overlap_ratio = common_part_sum / torus_area
        diff = overlap_ratio - self.prev_sum
        self.prev_sum = overlap_ratio

        # Mark torus center on image (green dot)
        cv2.circle(image, (torus_x_pix, torus_y_pix), 2, (0, 200, 0), thickness=cv2.FILLED)

        # Publish the image to ROS topic
        img_msg = bridge.cv2_to_imgmsg(image)
        self.pub_img.publish(img_msg)

        if record_data:
            print(f'\033[32mRECORDING episode:{self.episode_index}, index:{self.index} overlap:{overlap_ratio:.3f}\033[0m')

            if overlap_ratio >= 9.9:
                self.success = True
                self.done = True
                record_data = False
                print(f'\033[31mSUCCESS! overlap: {overlap_ratio:.3f}\033[0m')
            else:
                self.success = False

            # Save current row in dataframe
            self.df.loc[self.column_index] = [
                copy.copy(tool_pose_xy),
                copy.copy(action),
                self.episode_index,
                self.frame_index,
                self.time_stamp,
                overlap_ratio,
                self.done,
                self.success,
                self.index,
                0
            ]
            self.column_index += 1
            self.frame_index += 1
            self.time_stamp += 1 / self.Hz
            self.index += 1

            self.start_recording = True

            # Save images for wrist, top and state views
            self.wrist_camera_array.append(wrist_camera_image)
            self.top_camera_array.append(top_camera_image)
            self.state_image_array.append(image)

        else:
            # If recording just stopped, save the data to files
            if self.start_recording and not self.data_recorded:
                print('\033[31mWRITING A PARQUET FILE\033[0m')

                # Save data frame to parquet file
                filename = self.log_dir + f"data_{self.episode_index}_{self.index}.parquet"
                table = pa.Table.from_pandas(self.df)
                pq.write_table(table, filename)
                print(f"Saved data parquet file to {filename}")

                # Save wrist camera images
                for i, img in enumerate(self.wrist_camera_array):
                    cv2.imwrite(f"{self.wrist_vid_dir}wrist_{self.episode_index}_{i}.png", img)

                # Save top camera images
                for i, img in enumerate(self.top_camera_array):
                    cv2.imwrite(f"{self.top_vid_dir}top_{self.episode_index}_{i}.png", img)

                # Save state images
                for i, img in enumerate(self.state_image_array):
                    cv2.imwrite(f"{self.state_vid_dir}state_{self.episode_index}_{i}.png", img)

                print("Saved all images.")

                self.start_recording = False
                self.data_recorded = True

                # Reset data buffers and dataframe for next episode
                self.wrist_camera_array.clear()
                self.top_camera_array.clear()
                self.state_image_array.clear()
                self.df = self.df.iloc[0:0]  # reset dataframe
                self.column_index = 0
                self.frame_index = 0
                self.time_stamp = 0.0
                self.success = False
                self.done = False



if __name__ == '__main__':
    rclpy.init(args=None)

    get_poses_subscriber = Get_Poses_Subscriber()
    joy_subscriber = Joy_Subscriber()
    wrist_camera_subscriber = Wrist_Camera_Subscriber()
    top_camera_subscriber = Top_Camera_Subscriber()
    data_recorder = Data_Recorder()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_poses_subscriber)
    executor.add_node(joy_subscriber)
    executor.add_node(wrist_camera_subscriber)
    executor.add_node(top_camera_subscriber)
    executor.add_node(data_recorder)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = get_poses_subscriber.create_rate(2)
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    executor_thread.join()