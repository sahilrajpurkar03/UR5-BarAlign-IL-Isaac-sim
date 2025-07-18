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


        # mask for torus region, will be drawn as circle dynamically (for reward calculation)
        self.torus_region = np.zeros((self.initial_image.shape[0], self.initial_image.shape[1]), np.uint8)

        # filled image of torus on table
        self.torus_image = cv2.imread(os.environ['HOME'] + "/Rahul/UR5-BarAlign-RL-Isaac-sim/ur5_simulation/images/torus_top_plane_filled.png")
        self.torus_image = cv2.rotate(self.torus_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_gray = cv2.cvtColor(self.torus_image, cv2.COLOR_BGR2GRAY)
        thr, img_th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        self.blue_region = cv2.bitwise_not(img_th)
        self.blue_region_sum = cv2.countNonZero(self.blue_region)

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

        # Copy background image
        image = copy.copy(self.initial_image)

        self.torus_region[:] = 0

        # Convert torus and tool position to pixel coordinates
        torus_x_pix = int((1000 * torus_pose_xyw[0] + self.C_W) / self.scale)
        torus_y_pix = int((1000 * torus_pose_xyw[1] - self.C_H) / self.scale)
        tool_x_pix = int((1000 * tool_pose_xy[0] + self.C_W) / self.scale)
        tool_y_pix = int((1000 * tool_pose_xy[1] - self.C_H) / self.scale)

        # Draw tool circle (gray)
        cv2.circle(image, (tool_x_pix, tool_y_pix), self.radius, (100, 100, 100), thickness=cv2.FILLED)

        # Torus radii (adjust based on your object size)
        torus_outer_radius = int(110 / self.scale)       # outer ring radius
        ring_thickness = int(60 / self.scale)                            # thickness of the ring
        torus_inner_radius = torus_outer_radius - ring_thickness

        if torus_inner_radius < 0:
            torus_inner_radius = 0  # safety check

        # Create torus ring mask
        outer_mask = np.zeros_like(self.torus_region)
        inner_mask = np.zeros_like(self.torus_region)
        cv2.circle(outer_mask, (torus_x_pix, torus_y_pix), torus_outer_radius, 255, thickness=cv2.FILLED)
        cv2.circle(inner_mask, (torus_x_pix, torus_y_pix), torus_inner_radius, 255, thickness=cv2.FILLED)
        self.torus_region = cv2.bitwise_xor(outer_mask, inner_mask)

        # Create tool mask
        # tool_mask = np.zeros_like(self.torus_region)
        cv2.circle(image, (tool_x_pix, tool_y_pix), self.radius, 255, thickness=cv2.FILLED)

        # Overlap computation
        common_part = cv2.bitwise_and(self.blue_region, self.torus_region)
        common_part_sum = cv2.countNonZero(common_part)
        torus_sum = cv2.countNonZero(self.torus_region)

        # for DEBUG
        # print(f"[DEBUG] Common overlapping area (non-zero pixels): {common_part_sum}")
        # print(f"[DEBUG] Blue area (non-zero pixels): {self.blue_region_sum}")
        # print(f"[DEBUG] Torus area (non-zero pixels): {torus_sum}")

        # Save the mask to desktop (for debug)
        # desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        # torus_filename = os.path.join(desktop_path, "torus_mask.png")
        # blue_filename = os.path.join(desktop_path, "blue_mask.png")
        # common_filename = os.path.join(desktop_path, "common_mask.png")
        # cv2.imwrite(torus_filename, self.torus_region)
        # cv2.imwrite(blue_filename, self.blue_region)
        # cv2.imwrite(common_filename, common_part)
        # print(f"Saved torus mask to: {torus_filename}")
        # print(f"Saved blue mask to: {blue_filename}")
        # print(f"Saved common mask to: {common_filename}")

        overlap_ratio = common_part_sum/self.blue_region_sum
        sum_dif = overlap_ratio - self.prev_sum
        self.prev_sum = overlap_ratio

        # Draw torus ring and center with matching thickness to mask ring thickness
        torus_visual_outer_radius = int(85 / self.scale)
        torus_visual_thickness = int(55 / self.scale)
        cv2.circle(image, (torus_x_pix, torus_y_pix),  torus_visual_outer_radius, (0, 0, 180), thickness=torus_visual_thickness )
        cv2.circle(image, (torus_x_pix, torus_y_pix), 2, (0, 200, 0), thickness=cv2.FILLED)

        # Publish image to ROS
        img_msg = bridge.cv2_to_imgmsg(image)
        self.pub_img.publish(img_msg)

        if record_data:
            print(f'\033[32mRECORDING episode:{self.episode_index}, index:{self.index} overlap:{overlap_ratio:.3f}\033[0m')

            if overlap_ratio >= 0.90:
                self.success = True
                self.done = True
                record_data = False
                print(f'\033[31mSUCCESS! Overlap: {overlap_ratio:.3f}\033[0m')
            else:
                self.success = False

            # Save current step to DataFrame
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

            self.wrist_camera_array.append(wrist_camera_image)
            self.top_camera_array.append(top_camera_image)
            self.state_image_array.append(image)

        else:
            if self.start_recording and not self.data_recorded:
                print('\033[31mWRITING A PARQUET FILE\033[0m')

                if self.episode_index <= 9:
                    suffix = '00000' + str(self.episode_index)
                elif self.episode_index <= 99:
                    suffix = '0000' + str(self.episode_index)
                elif self.episode_index <= 999:
                    suffix = '000' + str(self.episode_index)
                elif self.episode_index <= 9999:
                    suffix = '00' + str(self.episode_index)
                else:
                    suffix = '0' + str(self.episode_index)

                data_file_name = f'episode_{suffix}.parquet'
                video_file_name = f'episode_{suffix}.mp4'

                table = pa.Table.from_pandas(self.df)
                pq.write_table(table, self.log_dir + data_file_name)
                print("Parquet file saved!")

                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out1 = cv2.VideoWriter(self.wrist_vid_dir + video_file_name, fourcc, self.Hz, (vid_W, vid_H))
                for frame1 in self.wrist_camera_array:
                    out1.write(frame1)
                out1.release()

                out2 = cv2.VideoWriter(self.top_vid_dir + video_file_name, fourcc, self.Hz, (vid_W, vid_H))
                for frame2 in self.top_camera_array:
                    out2.write(frame2)
                out2.release()

                out3 = cv2.VideoWriter(self.state_vid_dir + video_file_name, fourcc, self.Hz,
                                    (self.initial_image.shape[1], self.initial_image.shape[0]))
                for frame3 in self.state_image_array:
                    out3.write(frame3)
                out3.release()

                print("All videos saved!")
                self.data_recorded = True

    # Optional: Debug mask windows
    # cv2.imshow("Torus Region (Ring Mask)", self.torus_region)
    # cv2.imshow("Tool Mask", tool_mask)
    # cv2.imshow("Overlap", common_part)
    # cv2.waitKey(1)


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