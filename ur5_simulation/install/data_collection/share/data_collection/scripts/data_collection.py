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
tbar_pose_xyw = [0.0, 0.0, 0.0]
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
        global tool_pose_xy, tbar_pose_xyw

        # 0:tool
        tool_pose = data.transforms[0].transform.translation
        tool_pose_xy[0] = tool_pose.y
        tool_pose_xy[1] = tool_pose.x

        # 1:tbar
        tbar_translation  = data.transforms[1].transform.translation       
        tbar_rotation = data.transforms[1].transform.rotation 
        tbar_pose_xyw[0] = tbar_translation.y
        tbar_pose_xyw[1] = tbar_translation.x
        self.euler_angles[:] = R.from_quat([tbar_rotation.x, tbar_rotation.y, tbar_rotation.z, tbar_rotation.w]).as_euler('xyz', degrees=False)
        tbar_pose_xyw[2] = self.euler_angles[2]

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
        self.Hz = 10 # bridge data frequency
        self.prev_ee_pose = np.array([0, 0, 0], float)
        self.timer = self.create_timer(1/self.Hz, self.timer_callback)
        self.start_recording = False
        self.data_recorded = False

        #### log files for multiple runs are NOT overwritten
        base_dir = os.environ["HOME"] + "/Rahul/ur5_simulation/src/data_collection/scripts/my_pusht/"
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

        # image of a T shape on the table
        self.initial_image = cv2.imread(os.environ['HOME'] + "/Rahul/ur5_simulation/images/stand_top_plane.png")
        self.initial_image = cv2.rotate(self.initial_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
 
        # for reward calculation
        self.Tbar_region = np.zeros((self.initial_image.shape[0], self.initial_image.shape[1]), np.uint8)

        # filled image of T shape on the table
        self.T_image = cv2.imread(os.environ['HOME'] + "/Rahul/ur5_simulation/images/stand_top_plane_filled.png")
        self.T_image = cv2.rotate(self.T_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_gray = cv2.cvtColor(self.T_image, cv2.COLOR_BGR2GRAY)
        thr, img_th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        self.blue_region = cv2.bitwise_not(img_th)
        self.blue_region_sum = cv2.countNonZero(self.blue_region)
        # for debug
        #self.sum_img_pub = self.create_publisher(Image, '/sum_image', 10)
        
        self.pub_img = self.create_publisher(Image, '/pushT_image', 10)
        self.tool_radius = 10 # millimeters
        self.scale = 1.639344 # mm/pix
        self.C_W = 182 # pix
        self.C_H = 152 # pix
        self.OBL1 = int(150/self.scale)
        self.OBL2 = int(120/self.scale)
        self.OBW = int(30/self.scale)
        # radius of the tool
        self.radius = int(10/self.scale)

        self.df = pd.DataFrame(columns=['observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.reward', 'next.done', 'next.success', 'index', 'task_index'])
        self.index = 296 # 1 + the last index in the last episode
        self.episode_index = 1 # 1 + the last episode number
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
        global tool_pose_xy, tbar_pose_xyw, action, wrist_camera_image, top_camera_image, record_data
        
        image = copy.copy(self.initial_image)

        self.Tbar_region[:] = 0

        x = int((tool_pose_xy[0]*1000 + 300)/self.scale)
        y = int((tool_pose_xy[1]*1000 - 320)/self.scale)

        cv2.circle(image, center=(x, y), radius=self.radius, color=(100, 100, 100), thickness=cv2.FILLED)        
        
        # horizontal part of T
        x1 = tbar_pose_xyw[0]
        y1 = tbar_pose_xyw[1]
        th1 = -tbar_pose_xyw[2] - pi/2
        dx1 = -self.OBW/2*cos(th1 - pi/2)
        dy1 = -self.OBW/2*sin(th1 - pi/2)
        self.tbar1_ob = [[int(cos(th1)*self.OBL1/2     - sin(th1)*self.OBW/2   + dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*self.OBL1/2    + cos(th1)*self.OBW/2   + dy1 + (1000*y1-320)/self.scale)],
                          [int(cos(th1)*self.OBL1/2    - sin(th1)*(-self.OBW/2)+ dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*self.OBL1/2    + cos(th1)*(-self.OBW/2)+ dy1 + (1000*y1-320)/self.scale)],
                          [int(cos(th1)*(-self.OBL1/2) - sin(th1)*(-self.OBW/2)+ dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*(-self.OBL1/2) + cos(th1)*(-self.OBW/2)+ dy1 + (1000*y1-320)/self.scale)],
                          [int(cos(th1)*(-self.OBL1/2) - sin(th1)*self.OBW/2   + dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*(-self.OBL1/2) + cos(th1)*self.OBW/2   + dy1 + (1000*y1-320)/self.scale)]]  
        pts1_ob = np.array(self.tbar1_ob, np.int32)
        cv2.fillPoly(image, [pts1_ob], (0, 0, 180))
        cv2.fillPoly(self.Tbar_region, [pts1_ob], 255)
        
        #vertical part of T
        th2 = -tbar_pose_xyw[2] - pi
        dx2 = self.OBL2/2*cos(th2)
        dy2 = self.OBL2/2*sin(th2)
        self.tbar2_ob = [[int(cos(th2)*self.OBL2/2    - sin(th2)*self.OBW/2    + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2    + cos(th2)*self.OBW/2   + dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*self.OBL2/2    - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2    + cos(th2)*(-self.OBW/2)+ dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*(-self.OBL2/2) - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*(-self.OBW/2)+ dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*(-self.OBL2/2) - sin(th2)*self.OBW/2   + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*self.OBW/2   + dy2 + (1000*y1-320)/self.scale)]]  
        pts2_ob = np.array(self.tbar2_ob, np.int32)
        cv2.fillPoly(image, [pts2_ob], (0, 0, 180))
        cv2.fillPoly(self.Tbar_region, [pts2_ob], 255)

        common_part = cv2.bitwise_and(self.blue_region, self.Tbar_region)
        common_part_sum = cv2.countNonZero(common_part)
        sum = common_part_sum/self.blue_region_sum
        sum_dif = sum - self.prev_sum
        self.prev_sum = sum

        cv2.circle(image, center=(int(self.C_W + 1000*x1/self.scale), int((1000*y1-320)/self.scale)), radius=2, color=(0, 200, 0), thickness=cv2.FILLED)  

        img_msg = bridge.cv2_to_imgmsg(image)  
        self.pub_img.publish(img_msg) 

        # for debug
        #sum_image = bridge.cv2_to_imgmsg(common_part)  
        #self.sum_img_pub.publish(sum_image) 

        if record_data:
            print('\033[32m'+f'RECORDING episode:{self.episode_index}, index:{self.index} sum:{sum}'+'\033[0m')

            if sum >= 0.90:
                self.success = True
                self.done = True
                record_data = False
                print('\033[31m'+'SUCCESS!'+f': {sum}'+'\033[0m')
            else:
                self.success = False

            # lerobot/common/datasets/lerobot_dataset.py line 371~
            #                        ['observation.state', 'action', 'episode_index',     'frame_index', 'timestamp','next.reward','next.done','next.success','index', 'task_index']
            self.df.loc[self.column_index] = [copy.copy(tool_pose_xy), copy.copy(action), self.episode_index, self.frame_index, self.time_stamp, sum, self.done, self.success, self.index, 0]
            self.column_index += 1
            self.frame_index += 1
            self.time_stamp += 1/self.Hz
            self.index += 1

            self.start_recording = True

            self.wrist_camera_array.append(wrist_camera_image)
            self.top_camera_array.append(top_camera_image)
            self.state_image_array.append(image)

        else:
            if(self.start_recording and self.data_recorded == False):
                print('\033[31m'+'WRITING A PARQUET FILE'+'\033[0m')

                if self.episode_index <= 9:
                    data_file_name = 'episode_00000' + str(self.episode_index) + '.parquet'
                    video_file_name = 'episode_00000' + str(self.episode_index) + '.mp4'
                elif 9 < self.episode_index <= 99:
                    data_file_name = 'episode_0000' + str(self.episode_index) + '.parquet'
                    video_file_name = 'episode_0000' + str(self.episode_index) + '.mp4'
                elif 99 < self.episode_index <= 999:
                    data_file_name = 'episode_000' + str(self.episode_index) + '.parquet'
                    video_file_name = 'episode_000' + str(self.episode_index) + '.mp4'
                elif 999 < self.episode_index <= 9999:
                    data_file_name = 'episode_00' + str(self.episode_index) + '.parquet'
                    video_file_name = 'episode_00' + str(self.episode_index) + '.mp4'
                else:
                    data_file_name = 'episode_0' + str(self.episode_index) + '.parquet'
                    video_file_name = 'episode_0' + str(self.episode_index) + '.mp4'

                table = pa.Table.from_pandas(self.df)
                pq.write_table(table, self.log_dir + data_file_name)
                print("The parquet file is generated!")

                
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out1 = cv2.VideoWriter(self.wrist_vid_dir + video_file_name, fourcc, self.Hz, (vid_W, vid_H))
                for frame1 in self.wrist_camera_array:
                    out1.write(frame1)
                out1.release()
                print("The wrist video is generated!")
                out2 = cv2.VideoWriter(self.top_vid_dir + video_file_name, fourcc, self.Hz, (vid_W, vid_H))
                for frame2 in self.top_camera_array:
                    out2.write(frame2)
                out2.release()
                print("The top video is generated!")
                out3 = cv2.VideoWriter(self.state_vid_dir + video_file_name, fourcc, self.Hz, (self.initial_image.shape[1], self.initial_image.shape[0]))
                for frame3 in self.state_image_array:
                    out3.write(frame3)
                out3.release()
                print("The state video is generated!")

                self.data_recorded = True


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