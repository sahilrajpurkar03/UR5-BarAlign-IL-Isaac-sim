from pathlib import Path
#import gymnasium as gym
import os
import imageio
import numpy
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Image
import threading
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
from cv_bridge import CvBridge
import copy
from math import sin, cos, pi

bridge = CvBridge()

tool_pose_xy = [0.0, 0.0] # tool(end effector) pose
tbar_pose_xyw = [0.0, 0.0, 0.0]
vid_H = 360
vid_W = 640
wrist_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
top_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Create a directory to store the video of the evaluation
#output_directory = Path("outputs/eval/example_pusht_diffusion")
#output_directory.mkdir(parents=True, exist_ok=True)

# Select your device
device = "cuda"

# Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
# pretrained_policy_path = "lerobot/diffusion_pusht"
# OR a path to a local outputs/train folder.
pretrained_policy_path = Path("outputs/train/my_pusht_diffusion/20250329093535")

policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.

# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print(policy.config.input_features)
#print(env.observation_space)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print(policy.config.output_features)
#print(env.action_space)

# Reset the policy and environments to prepare for rollout
policy.reset()
#numpy_observation, info = env.reset(seed=42)

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
#frames.append(env.render())

step = 0
done = False

class Get_End_Effector_Pose(Node):

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

class Action_Publisher(Node):

    def __init__(self):
        super().__init__('Joy_Publisher')
        Hz = 10 # publish rate
        
        self.pub_joy = self.create_publisher(Joy, '/joy', 10)
        self.joy_commands = Joy()
        self.joy_commands.axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.joy_commands.buttons = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.timer = self.create_timer(1/Hz, self.timer_callback)

        # image of a T shape on the table
        self.initial_image = cv2.imread(os.environ['HOME'] + "/ur5_simulation/images/stand_top_plane.png")
        self.initial_image = cv2.rotate(self.initial_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

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

        self.top_camera_array = []
        self.state_image_array = []

    def timer_callback(self):
        global tool_pose_xy, tbar_pose_xyw, action, wrist_camera_image, top_camera_image

        self.joy_commands.header.frame_id = "joy"
        self.joy_commands.header.stamp = joy_publisher.get_clock().now().to_msg()
        
        base_image = copy.copy(self.initial_image)
        #print(f"tbar_pose_xyw:{tbar_pose_xyw}")
        #self.Tbar_region[:] = 0

        x = int((tool_pose_xy[0]*1000 + 300)/self.scale)
        y = int((tool_pose_xy[1]*1000 - 320)/self.scale)
        #print(f"x, y:{x}, {y}")
        cv2.circle(base_image, center=(x, y), radius=self.radius, color=(100, 100, 100), thickness=cv2.FILLED)        
        
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
        cv2.fillPoly(base_image, [pts1_ob], (0, 0, 180))
        #cv2.fillPoly(self.Tbar_region, [pts1_ob], 255)
        
        #vertical part of T
        th2 = -tbar_pose_xyw[2] - pi
        dx2 = self.OBL2/2*cos(th2)
        dy2 = self.OBL2/2*sin(th2)
        self.tbar2_ob = [[int(cos(th2)*self.OBL2/2    - sin(th2)*self.OBW/2    + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2    + cos(th2)*self.OBW/2   + dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*self.OBL2/2    - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2    + cos(th2)*(-self.OBW/2)+ dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*(-self.OBL2/2) - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*(-self.OBW/2)+ dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*(-self.OBL2/2) - sin(th2)*self.OBW/2   + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*self.OBW/2   + dy2 + (1000*y1-320)/self.scale)]]  
        pts2_ob = np.array(self.tbar2_ob, np.int32)
        cv2.fillPoly(base_image, [pts2_ob], (0, 0, 180))
        #cv2.fillPoly(self.Tbar_region, [pts2_ob], 255)

        '''
        common_part = cv2.bitwise_and(self.blue_region, self.Tbar_region)
        common_part_sum = cv2.countNonZero(common_part)
        sum = common_part_sum/self.blue_region_sum
        sum_dif = sum - self.prev_sum
        self.prev_sum = sum
        #print(f"common_part %:{sum_dif}")
        '''

        cv2.circle(base_image, center=(int(self.C_W + 1000*x1/self.scale), int((1000*y1-320)/self.scale)), radius=2, color=(0, 200, 0), thickness=cv2.FILLED)  

        img_msg = bridge.cv2_to_imgmsg(base_image)  
        self.pub_img.publish(img_msg) 

        # Prepare observation for the policy running in Pytorch
        print(f"tool_pose_xy:{tool_pose_xy}")
        state = torch.from_numpy(np.array(tool_pose_xy))
        image = torch.from_numpy(base_image)

        # Convert to float32 with image from channel first in [0,255]
        # to channel last in [0,1]
        state = state.to(torch.float32)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        image = image.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.image": image,
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        print(f"numpy_action:{numpy_action}")
        self.joy_commands.axes[0] = numpy_action[0]
        self.joy_commands.axes[1] = numpy_action[1]

        self.pub_joy.publish(self.joy_commands)

if __name__ == '__main__':
    rclpy.init(args=None)

    get_end_effector_pose = Get_End_Effector_Pose()
    joy_publisher = Action_Publisher()
    #wrist_camera_subscriber = Wrist_Camera_Subscriber()
    #top_camera_subscriber = Top_Camera_Subscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_end_effector_pose)
    executor.add_node(joy_publisher)
    #executor.add_node(wrist_camera_subscriber)
    #executor.add_node(top_camera_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = joy_publisher.create_rate(2)
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    executor_thread.join()