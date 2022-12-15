import os
import string
import cv2
import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import logging
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, StaticObject, ScenarioObject
from beamngpy.sensors import Camera, GForces, Electrics, Damage, Timer

import scipy.misc
import copy
import torch
import torchvision.transforms as transforms
import statistics, math
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import PIL
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'''
Actor-Critic Learner with continuous action space
actor-critic model based on Microsoft example:
https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/6-Other/22-DeepRL/CartPole-RL-TF.ipynb
DQN from microsoft example:
https://microsoft.github.io/AirSim/reinforcement_learning/
'''

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

import gym
from gym import spaces
# from airgym.envs.airsim_env import AirSimEnv


class CarEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape=(3, 135, 240), model="DQN", filepathroot=".", beamngpath="C:/Users/Meriel/Documents",
                 beamnginstance="BeamNG.research", port=64356, scenario="west_coast_usa", road_id="12146", reverse=False,
                 base_model=None, test_model=False):
        super(CarEnv, self).__init__()
        self.test_model = test_model
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,1), dtype=np.float32)
        self.transform = T.Compose([T.ToTensor()])
        self.episode_steps, self.frames_adjusted = 0, 0
        self.beamngpath = beamngpath
        if base_model is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.base_model = torch.load(base_model, map_location=device).eval() #torch.load(base_model).to(device)
        self.default_scenario = scenario
        self.road_id = road_id
        self.reverse = reverse
        self.integral = 0.0
        self.prev_error = 0.0
        self.overall_throttle_setpoint = 40
        self.setpoint = self.overall_throttle_setpoint
        self.lanewidth = 3.75  # 2.25
        self.track_cam_pos = None
        self.centerline = []
        self.centerline_interpolated = []
        self.roadleft = []
        self.roadright = []
        self.trajectories, self.current_trajectory = [], []
        self.all_rewards, self.current_rewards = [], []
        self.actions, self.base_model_inf = [], []
        self.image_shape = image_shape
        self.steps_per_sec = 30
        self.runtime = 0.0
        self.episode = 0
        random.seed(1703)
        setup_logging()
        self.model = model
        self.deflation_pattern = filepathroot
        beamng = BeamNGpy('localhost', port=port, home=f'{self.beamngpath}/BeamNG.research.v1.7.0.1', user=f'{self.beamngpath}/{beamnginstance}')

        self.scenario = Scenario(self.default_scenario, 'RL_Agent-test')
        self.vehicle = Vehicle('ego_vehicle', model="hopper", licence='EGO', color="green")

        self.setup_sensors()
        self.spawn = self.spawn_point()

        self.scenario.add_vehicle(self.vehicle, pos=self.spawn['pos'], rot=None, rot_quat=self.spawn['rot_quat'])

        # setup free camera
        cam_pos = (973.684363136209, -854.1390707377659,
                   self.spawn['pos'][2] + 500)
        eagles_eye_cam = Camera(cam_pos, #(0.013892743289471, -0.015607489272952, -1.39813470840454, 0.91656774282455),
                                (self.spawn['rot_quat'][0], self.spawn['rot_quat'][1], -1.39813470840454, 0.91656774282455),
                                fov=90, resolution=(1500, 1500),
                                colour=True, depth=False, annotation=False)
        self.scenario.add_camera(eagles_eye_cam, "eagles_eye_cam")

        self.scenario.make(beamng)
        self.bng = beamng.open(launch=True)

        self.bng.set_deterministic()
        self.bng.set_steps_per_second(self.steps_per_sec)
        self.bng.load_scenario(self.scenario)

        print("Interpolating centerline...")
        line = self.create_ai_line_from_road_with_interpolation()

        print("Starting scenario....")
        self.bng.start_scenario()
        print("Pausing BeamNG...")
        self.bng.pause()

        print(f"midpoint={self.centerline_interpolated[int(len(self.centerline_interpolated)/2)]}")

        freecams = self.scenario.render_cameras()
        freecams['eagles_eye_cam']["colour"].convert('RGB').save(f"eagles-eye-view-{self.default_scenario}-{self.road_id}.jpg", "JPEG")
        assert self.vehicle.skt
        ###########################################################################
        self.observation_space = spaces.Box(0, 255, shape=self.image_shape, dtype=np.uint8)
        self.viewer = None

        self.start_ts = 0
        self.state = {
            "image": np.zeros(image_shape[1:]),
            "prev_image": np.zeros(image_shape[1:]),
            "pose": np.zeros(3),
            "prev_pose": np.zeros(3),
            "collision": False,
        }

        self.image = None
        self.car_state = None


    def _get_obs(self):
        self.state["prev_pose"] = self.vehicle.state["pos"]
        self.vehicle.update_vehicle()
        sensors = self.bng.poll_sensors(self.vehicle)
        self.current_trajectory.append(self.vehicle.state["pos"])
        # responses = self.car.simGetImages([self.image_request])
        self.car_state = sensors
        self.state["pose"] = self.vehicle.state["pos"]
        kph = self.ms_to_kph(sensors['electrics']['wheelspeed'])
        self.state["collision"] = sensors["damage"]["damage"] > 0
        image = np.array(sensors['front_cam']['colour'].convert('RGB'), dtype=np.uint8)
        image = cv2.resize(image, (self.image_shape[3], self.image_shape[2]))
        if self.image_shape[1] == 1:
            image = self.rgb2gray(image)
        image = image.reshape(self.image_shape[1:])
        self.state["prev_image"] = self.state["image"]
        self.state["image"] = image
        # outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg = self.has_car_left_track()
        # expert_action, cartocl_theta_deg = self.get_expert_action(outside_track, distance_from_center,
        #                                                           leftrightcenter, segment_shape, theta_deg)
        # evaluation = self.evaluator(outside_track, distance_from_center, leftrightcenter, segment_shape)
        # return {"image1": image, "image2": self.state["prev_image"], "kph": kph}
        combo = np.zeros((self.image_shape))
        combo[0] = image
        combo[1] = self.state["prev_image"]
        # print(f"{combo.shape=}")
        if self.episode_steps == 0:
            return np.zeros(self.image_shape)
        else:
            return combo

    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return np.array(gray, dtype=np.uint8)

    def step(self, action):
        if self.test_model:
            obs = self._get_obs()
            kph = self.ms_to_kph(self.car_state['electrics']['wheelspeed'])
            dt = (self.car_state['timer']['time'] - self.start_time) - self.runtime
            self.runtime = self.car_state['timer']['time'] - self.start_time
            image = np.array(self.car_state['front_cam']['colour'].convert('RGB'), dtype=np.uint8)
            image = cv2.resize(image, (self.image_shape[3], self.image_shape[2]))
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("true image", img)
            cv2.waitKey(1)

            features = self.transform(image)[None]
            base_model_inf = self.base_model(features).item()
            self.base_model_inf.append(base_model_inf)
            outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg = self.has_car_left_track()
            if abs(action.item() - base_model_inf) < 0.005:
                taken_action = base_model_inf
                blackedout = np.zeros(image.shape)
                cv2.imshow("action image", blackedout)
                cv2.waitKey(1)
            else:
                taken_action = action.item()
                blackedout = np.ones(image.shape)
                blackedout[:,:,:2] = blackedout[:,:,:2] * 0 # RED
                cv2.imshow("action image", blackedout)
                cv2.waitKey(1)
                self.frames_adjusted += 1

            if abs(taken_action) > 0.15:
                self.setpoint = 30
            else:
                self.setpoint = 40
            throttle = self.throttle_PID(kph, dt)
            self.vehicle.control(throttle=throttle, steering=taken_action, brake=0.0)
            self.bng.step(1, wait=True)
            outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg = self.has_car_left_track()
            # reward = self.calc_reward(action, expert_action, self.evaluator(outside_track, distance_from_center, leftrightcenter, segment_shape))
            reward = 0

            self.episode_steps += 1
            print(f'{outside_track=}, {self.state["collision"]=}')
            done = outside_track or self.state["collision"]
            self.current_rewards.append(reward)
            # print(f"STEP() \n\troad_seg {theta_deg=:.1f}\t car-to-CL theta_deg={cartocl_theta_deg:.1f}\n\texpert_action={expert_action:.3f}\t\t{base_model_inf=:.3f}\n\t{reward=:.1f}\n\t{done=}\t{outside_track=}\tcollision={self.state['collision']}")
            return obs, reward, done, self.state

        else:
            obs = self._get_obs()
            kph = self.ms_to_kph(self.car_state['electrics']['wheelspeed'])
            dt = (self.car_state['timer']['time'] - self.start_time) - self.runtime
            self.runtime = self.car_state['timer']['time'] - self.start_time
            image = np.array(self.car_state['front_cam']['colour'].convert('RGB'), dtype=np.uint8)
            image = cv2.resize(image, (self.image_shape[3], self.image_shape[2]))
            cv2.imshow("true image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            features = self.transform(image)[None]
            base_model_inf = self.base_model(features).item()
            self.base_model_inf.append(base_model_inf)
            outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg = self.has_car_left_track()
            expert_action, cartocl_theta_deg = self.get_expert_action(outside_track, distance_from_center,
                                                                      leftrightcenter, segment_shape, theta_deg)
            evaluation = self.evaluator(outside_track, distance_from_center, leftrightcenter, segment_shape)
            taken_action = None
            if evaluation == 1:
                taken_action = base_model_inf
                # 93275.319|E|libbeamng.lua.V.updateGFX|Object ID: 9620
                # 93275.320|E|libbeamng.lua.V.updateGFX|Object position: vec3(601.547,53.4482,43.29)
                # 93275.320|E|libbeamng.lua.V.updateGFX|Object rotation: quat(-0.033713240176439,0.038085918873549,0.99870544672012,-0.00058328913291916)

                # 93467.556|E|libbeamng.lua.V.updateGFX|Object ID: 9620
                # 93467.557|E|libbeamng.lua.V.updateGFX|Object position: vec3(491.521,119.336,30.2536)
                # 93467.557|E|libbeamng.lua.V.updateGFX|Object rotation: quat(-0.062928266823292,0.038666535168886,0.68600910902023,0.72383457422256)

                if abs(base_model_inf) > 0.15:
                    self.setpoint = 30
                else:
                    self.setpoint = 40
                throttle = self.throttle_PID(kph, dt)
                cv2.imshow("action image", np.zeros(image.shape)) # black
                cv2.waitKey(1)

                self.vehicle.control(throttle=throttle, steering=base_model_inf, brake=0.0)

            else:
                taken_action = expert_action
                blackedout = np.ones(image.shape)
                blackedout[:,:,:2] = blackedout[:,:,:2] * 0
                cv2.imshow("action image", blackedout) # red
                cv2.waitKey(1)
                if abs(expert_action) > 0.15:
                    self.setpoint = 30
                else:
                    self.setpoint = 40

                throttle = self.throttle_PID(kph, dt)
                self.vehicle.control(throttle=throttle, steering=expert_action, brake=0.0)
                self.frames_adjusted += 1

            self.bng.step(1, wait=True)
            outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg = self.has_car_left_track()
            evaluation = self.evaluator(outside_track, distance_from_center, leftrightcenter, segment_shape)
            reward = self.calc_reward(action, taken_action, evaluation)
            self.episode_steps += 1
            done = outside_track or self.state["collision"] or (self.distance2D(self.state["pose"], [601.547,53.4482,43.29]) < 20)
            self.current_rewards.append(reward)
            # print(f"STEP() \n\troad_seg {theta_deg=:.1f}\t car-to-CL theta_deg={cartocl_theta_deg:.1f}\n\texpert_action={expert_action:.3f}\t\t{base_model_inf=:.3f}\n\t{reward=:.1f}\n\t{done=}\t{outside_track=}\tcollision={self.state['collision']}")
            return obs, reward, done, self.state


    def calc_reward(self, action, expert_action, evaluation):
        reward = 0
        # conf = (action[0][0] + 1) / 2
        # if round(conf, 0) == evaluation:
        #     reward += 50
        # else:
        #     reward -= 50
        # print(f"{action=} \t conf={round(conf, 0)} \t {evaluation=} \t conf. reward={reward} \t action reward={-(abs(expert_action - action[0][1])*100):.3f}")
        # reward -= (abs(expert_action - action[0][1])*100)
        # print(f"{action=} \t {reward=}")
        if round(abs(expert_action - action.item()), 3) < 0.005:
            reward += 100
        else:
            reward -= (abs(expert_action - action.item())*100)
        if self.state["collision"]:
            reward -= 10000
        return reward


    def reset(self):
        print(f"\n\n\nRESET()")
        if self.episode > 0:
            print(f"SUMMARY OF EPISODE #{self.episode}")
            self.trajectories.append(self.current_trajectory)
            self.all_rewards.append(sum(self.current_rewards))
            dist = self.get_distance_traveled(self.current_trajectory)
            dist_from_centerline = []
            for i in self.current_trajectory:
                distance_from_centerline = self.dist_from_line(self.centerline_interpolated, i)
                dist_from_centerline.append(min(distance_from_centerline))
            # print(f"\ttotal distance travelled:{dist}\n\ttotal reward:{sum(self.current_rewards)}\n\tavg dist from centerline:{sum(dist_from_centerline) / len(dist_from_centerline)}")
            print(f"\ttotal distance travelled:{dist}"
                  f"\n\ttotal episode reward:{sum(self.current_rewards)}"
                  f"\n\tavg dist from centerline:{sum(dist_from_centerline) / len(dist_from_centerline)}"
                  f"\n\tpercent frames adjusted:{self.frames_adjusted / self.episode_steps}")
            # self.plot_deviation(f"{self.model} {dist=:.1f} ep={self.episode} start", self.deflation_pattern, start_viz=True)
            self.plot_deviation(f"{self.model} {dist=:.1f} ep={self.episode}", self.deflation_pattern+str(f" rew={sum(self.current_rewards)}"), start_viz=False)
            self.plot_durations(self.all_rewards, save=True, title=self.deflation_pattern)
        self.episode += 1
        self.episode_steps, self.frames_adjusted = 0, 0
        self.current_trajectory = []
        self.current_rewards = []
        self.integral, self.prev_error = 0.0, 0.0
        self.bng.restart_scenario()
        self.bng.step(1, wait=True)
        self.vehicle.update_vehicle()
        sensors = self.bng.poll_sensors(self.vehicle)
        # spawnpoint = [290.558, -277.28, 46.0]
        # endpoint = self.actual_middle[-1]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.prev_error = self.setpoint
        self.overall_damage, self.runtime, self.wheelspeed, self.distance_from_center = 0.0, 0.0, 0.0, 0.0
        self.total_loops, self.total_imgs, self.total_predictions = 0, 0, 0
        self.start_time = sensors['timer']['time']
        self.image = np.array(sensors['front_cam']['colour'].convert('RGB'))
        self.image = cv2.resize(self.image, (self.image_shape[3], self.image_shape[2]))
        # image = np.array(sensors['front_cam']['colour'].convert('RGB'))
        # self.image = self.rgb2gray(self.image).reshape(self.image_shape)
        # print(f"{self.image.shape=}")
        if self.image_shape[1] == 1:
            self.image = self.rgb2gray(self.image)
        self.image = self.image.reshape(self.image_shape[1:])
        self.outside_track = False
        self.done = False
        self.states, self.actions, self.probs, self.rewards, self.critic_values = [], [], [], [], []
        self.traj = []
        obs = self._get_obs()
        # print(f"reset {obs.shape=}")
        return obs

    # def render(self):
    #     return self._get_obs()

    # bad = 0, good = 1
    def evaluator(self, outside_track, distance_from_center, leftrightcenter, segment_shape, steer=None):
        if steer is None:
            steer = self.base_model_inf[-1]
        if leftrightcenter == 0 and abs(steer) <= 0.15 and segment_shape == 0:
            # centered, driving straight, straight road
            # print(f"EVAL(base_model_inf={steer:.3f})=GOOD \n\tcentered, driving straight, straight road")
            return 1
        elif leftrightcenter == 0 and steer < -0.15 and segment_shape == 1:
            # centered, driving left, left curve road
            # print(f"EVAL(base_model_inf={steer:.3f})=GOOD\n\tcentered, driving left, left curve road")
            return 1
        elif leftrightcenter == 0 and steer > 0.15 and segment_shape == 2:
            # centered, driving right, right curve road
            # print(f"EVAL(base_model_inf={steer:.3f})=GOOD\n\tcentered, driving right, right curve road")
            return 1
        elif leftrightcenter == 1 and steer > 0.15:
            # left of center, turning right
            # print(f"EVAL(base_model_inf={steer:.3f})=GOOD\n\tleft of center, turning right")
            return 1
        elif leftrightcenter == 2 and steer < -0.15:
            # right of center, turning left
            # print(f"EVAL(base_model_inf={steer:.3f})=GOOD\n\tright of center, turning left")
            return 1
        else:
            # print(f"EVAL(base_model_inf={steer:.3f})=BAD \t{leftrightcenter=}\t{segment_shape=}")
            return 0

    def get_expert_action(self, outside_track, distance_from_center, leftrightcenter, segment_shape, theta_deg):
        distance_from_centerline = self.dist_from_line(self.centerline_interpolated, self.vehicle.state['pos'])
        dist = min(distance_from_centerline)
        i = np.where(distance_from_centerline == dist)[0][0]
        next_point = self.centerline_interpolated[(i + 3) % len(self.centerline_interpolated)]
        # print(f"{i=}\t{i+3=}\t{len(self.centerline_interpolated)=}")
        theta_deg = self.get_angle_between_3_points_atan2(self.vehicle.state['pos'][0:2], next_point[0:2], self.vehicle.state['front'][0:2])
        action = theta_deg / 180
        return action, theta_deg

    # track ~12.50m wide; car ~1.85m wide
    def has_car_left_track(self):
        self.vehicle.update_vehicle()
        vehicle_pos = self.vehicle.state['front'] #self.vehicle.state['pos']
        distance_from_centerline = self.dist_from_line(self.centerline_interpolated, vehicle_pos)
        dist = min(distance_from_centerline)
        i = np.where(distance_from_centerline == dist)[0][0]
        leftrightcenter = self.get_position_relative_to_centerline(dist, i, centerdist=1.5)
        segment_shape, theta_deg = self.get_current_segment_shape(vehicle_pos)
        return dist > 4.0, dist, leftrightcenter, segment_shape, theta_deg


    def get_current_segment_shape(self, vehicle_pos):
        distance_from_centerline = self.dist_from_line(self.actual_middle, vehicle_pos)
        dist = min(distance_from_centerline)
        i = np.where(distance_from_centerline == dist)[0][0]
        A = np.array(self.actual_middle[(i + 2) % len(self.actual_middle)])
        B = np.array(self.actual_middle[i])
        C = np.array(self.roadright[i])
        # try:
        theta = math.acos(np.vdot(B-A, B-C) / (np.linalg.norm(B-A) * np.linalg.norm(B-C)))
        # except ValueError:
        #     # print(f"{A=}\t{B=}\t{C=}")
        #     theta = 0
        theta_deg = math.degrees(theta)
        # print(f"{math.degrees(theta)=:.1f}")
        if theta_deg > 110:
            # print(f"Road curving left \t{theta_deg=:.1f}")
            return 1, theta_deg
        elif theta_deg < 70:
            # print(f"Road curving right \t{theta_deg=:.1f}")
            return 2, theta_deg
        else:
            # print(f"Road is straight \t{theta_deg=:.1f}")
            return 0, theta_deg


    def get_angle_between_3_points_atan2(self, A, B, C):
        result = math.atan2(C[1] - A[1], C[0] - A[0]) - math.atan2(B[1] - A[1], B[0] - A[0])
        result = math.degrees(result)
        if result > 180:
            return result - 360
        elif result < -180:
            return result + 360
        return result


    # solve for gamma where a is the corresponding vertex of gamma
    def law_of_cosines(self, A, B, C):
        dist_AB = self.distance2D(A[:2], B[:2])
        dist_BC = self.distance2D(B[:2], C[:2])
        dist_AC = self.distance2D(A[:2], C[:2])
        arccos = math.acos((math.pow(dist_AB, 2) + math.pow(dist_AC, 2) - math.pow(dist_BC, 2)) / (2 * dist_AB * dist_AC))
        return math.degrees(arccos)


    def get_angle_between_3_points(self, A,B,C):
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        x = np.vdot(B-A, B-C)
        y = np.linalg.norm(B-A)
        z = np.linalg.norm(B-C)
        cosine = x / (y * z)
        theta = math.acos(cosine)
        # print(f"{math.degrees(theta)=:.1f}")
        return math.degrees(theta)


    def distance2D(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_position_relative_to_centerline(self, dist, i, centerdist=1):
        A = self.centerline_interpolated[i]
        B = self.centerline_interpolated[(i + 2) % len(self.centerline_interpolated)]
        P = self.vehicle.state['front']
        d = (P[0]-A[0])*(B[1]-A[1])-(P[1]-A[1])*(B[0]-A[0])
        if abs(dist) < centerdist:
            # print(f"CENTER, {dist=:.1f} {d=:.1f}")
            return 0
        elif d < 0:
            # print(f"LEFT, {dist=:.1f} {d=:.1f}")
            return 1
        elif d > 0:
            # print(f"RIGHT, {dist=:.1f} {d=:.1f}")
            return 2

    ################################# BEAMNG HELPERS #################################

    def setup_sensors(self):
        camera_pos = (-0.5, 0.38, 1.3)
        camera_dir = (0, 1.0, 0)
        fov = 51 # 60 works for full lap #63 breaks on hairpin turn
        width = int(self.image_shape[3] / 2)
        height = int(self.image_shape[2] / 2)
        resolution = (width, height)
        front_camera = Camera(camera_pos, camera_dir, fov, resolution,
                              colour=True, depth=True, annotation=True)
        gforces = GForces()
        electrics = Electrics()
        damage = Damage()
        timer = Timer()
        self.vehicle.attach_sensor('front_cam', front_camera)
        self.vehicle.attach_sensor('gforces', gforces)
        self.vehicle.attach_sensor('electrics', electrics)
        self.vehicle.attach_sensor('damage', damage)
        self.vehicle.attach_sensor('timer', timer)

    def spawn_point(self):
        lanewidth = 2.75
        if self.default_scenario == 'cliff':
            #return {'pos':(-124.806, 142.554, 465.489), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
            return {'pos': (-124.806, 190.554, 465.489), 'rot': None, 'rot_quat': (0, 0, 0.3826834, 0.9238795)}
        elif self.default_scenario == 'west_coast_usa':
            if self.road_id == 'midhighway': # mid highway scenario (past shadowy parts of road)
                return {'pos': (-145.775, 211.862, 115.55), 'rot': None, 'rot_quat': (0.0032586499582976, -0.0018308814615011, 0.92652350664139, -0.37621837854385)}
            # actually past shadowy parts of road?
            #return {'pos': (95.1332, 409.858, 117.435), 'rot': None, 'rot_quat': (0.0077012465335429, 0.0036200874019414, 0.90092438459396, -0.43389266729355)}
            # surface road (crashes early af)
            elif self.road_id == "13242":
                return {'pos': (-733.7, -923.8, 163.9), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, 0.805, 0.592), -20)}
            elif self.road_id == "8650":
                return {'pos': (-365.24, -854.45, 136.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 90)}
            elif self.road_id == "12667":
                return {'pos': (-892.4, -793.4, 114.1), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 70)}
            elif self.road_id == "8432":
                if self.reverse:
                    return {'pos': (-871.9, -803.2, 115.3), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 165 - 180)}
                return {'pos': (-390.4,-799.1,139.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 165)}
            elif self.road_id == "8518":
                if self.reverse:
                    return {'pos': (-913.2, -829.6, 118.0), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 60)}
                return {'pos': (-390.5, -896.6, 138.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 20)}
            elif self.road_id == "8417":
                return {'pos': (-402.7,-780.2,141.3), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            elif self.road_id == "8703":
                if self.reverse:
                    return {'pos': (-312.4, -856.8, 135.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
                return {'pos': (-307.8,-784.9,137.6), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 80)}
            elif self.road_id == "12641":
                if self.reverse:
                    return {'pos': (-964.2, 882.8, 75.1), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
                return {'pos': (-366.1753845214844, 632.2236938476562, 75.1), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 180)}
            elif self.road_id == "13091":
                if self.reverse:
                    return {'pos': (-903.6078491210938, -586.33154296875, 106.6), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 80)}
                return {'pos': (-331.0728759765625,-697.2451782226562,133.0), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            # elif self.road_id == "11602":
            #     return {'pos': (-366.4, -858.8, 136.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            elif self.road_id == "12146":
                if self.reverse:
                    return {'pos': (995.7, -855.0, 167.1), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -15)}
                return {'pos': (-391.0,-798.8, 139.7), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -15)}
            elif self.road_id == "13228":
                return {'pos': (-591.5175170898438,-453.1298828125,114.0), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            elif self.road_id == "13155": #middle laneline on highway #12492 11930, 10368 is an edge
                return {'pos': (-390.7796936035156, -36.612098693847656, 109.9), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -90)}
            elif self.road_id == "10784": # 13228  suburb edge, 12939 10371 12098 edge
                return {'pos': (57.04786682128906, -150.53302001953125, 125.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -115)}
            elif self.road_id == "10673":
                return {'pos': (-21.712169647216797, -826.2122802734375, 133.1), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -90)}
            elif self.road_id == "12930": # 13492 dirt road
                return {'pos': (-347.16302490234375,-824.6746215820312,137.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -100)}
            elif self.road_id == "10988":
                return {'pos': (622.163330078125,-251.1154022216797,146.99), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -90)}
            elif self.road_id == "13306":
                return {'pos': (-310,-790.044921875,137.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), -30)}
            elif self.road_id == "13341":
                return {'pos': (-393.4385986328125,-34.0107536315918,109.64727020263672), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 90)}
            # elif self.road_id == "":
            #     return {'pos': (), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            # elif self.road_id == "":
            #     return {'pos': (), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            # elif self.road_id == "":
            #     return {'pos': (), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            # elif self.road_id == "":
            #     return {'pos': (), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            # elif self.road_id == "":
            #     return {'pos': (), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.278, 0.961), 0)}
            elif self.road_id == '12669':
                return {'pos': (456.85526276, -183.39646912,  145.54124832), 'rot': None, 'rot_quat': self.turn_X_degrees((-0.043629411607981, 0.021309537813067, 0.98556911945343, 0.16216005384922), 90)}
            elif self.road_id == 'surfaceroad1':
                return {'pos': (945.285, 886.716, 132.061), 'rot': None, 'rot_quat': (-0.043629411607981, 0.021309537813067, 0.98556911945343, 0.16216005384922)}
            # surface road 2
            elif self.road_id == 'surfaceroad2':
                return {'pos': (900.016, 959.477, 127.227), 'rot': None, 'rot_quat': (-0.046136282384396, 0.018260028213263, 0.94000166654587, 0.3375423848629)}
            # surface road 3 (start at top of hill)
            elif self.road_id == 'surfaceroad3':
                return {'pos':(873.494, 984.636, 125.398), 'rot': None, 'rot_quat':(-0.043183419853449, 2.3034785044729e-05, 0.86842048168182, 0.4939444065094)}
            # surface road 4 (right turn onto surface road) (HAS ACCOMPANYING AI DIRECTION AS ORACLE)
            elif self.road_id == 'surfaceroad4':
                return {'pos': (956.013, 838.735, 134.014), 'rot': None, 'rot_quat': (0.020984912291169, 0.037122081965208, -0.31912142038345, 0.94675397872925)}
            # surface road 5 (ramp past shady el)
            elif self.road_id == 'surfaceroad5':
                return {'pos':(166.287, 812.774, 102.328), 'rot': None, 'rot_quat':(0.0038638345431536, -0.00049926445353776, 0.60924011468887, 0.79297626018524)}
            # entry ramp going opposite way
            elif self.road_id == 'entryrampopp':
                return {'pos': (850.136, 946.166, 123.827), 'rot': None, 'rot_quat': (-0.030755277723074, 0.016458060592413, 0.37487033009529, 0.92642092704773)}
            # racetrack
            elif self.road_id == 'racetrack':
                return {'pos': (395.125, -247.713, 145.67), 'rot': None, 'rot_quat': (0, 0, 0.700608, 0.713546)}
        elif self.default_scenario == 'smallgrid':
            return {'pos':(0.0, 0.0, 0.0), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
            # right after toll
            return {'pos': (-852.024, -517.391 + lanewidth, 106.620), 'rot': None, 'rot_quat': (0, 0, 0.926127, -0.377211)}
            # return {'pos':(-717.121, 101, 118.675), 'rot':None, 'rot_quat':(0, 0, 0.3826834, 0.9238795)}
            return {'pos': (-717.121, 101, 118.675), 'rot': None, 'rot_quat': (0, 0, 0.918812, -0.394696)}
        elif self.default_scenario == 'automation_test_track':
            if self.road_id == 'startingline':
                # starting line
                return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
            elif self.road_id == "7991":
                return {'pos': (57.229, 360.560, 128.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.702719, 0.711467), 180)}
            elif self.road_id == "8293":
                return {'pos': (-556.185, 386.985, 145.5), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.702719, 0.711467), 120)}
            elif self.road_id == '8185': # highway (open, farm-like)
                # return {'pos': (174.9,-289.7,120.8), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.704635, 0.70957), 180)}
                return {'pos': (36.742,-269.105,120.461), 'rot': None, 'rot_quat': (-0.0070,0.0082,0.7754,0.6314)}
                # return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.704635, 0.70957), 180)}
            elif self.road_id == 'starting line 30m down':
                # 30m down track from starting line
                return {'pos': (530.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
            elif self.road_id == 'handlingcircuit':
                # handling circuit
                return {'pos': (-294.031, 10.4074, 118.518), 'rot': None, 'rot_quat': (0, 0, 0.708103, 0.706109)}
            elif self.road_id == 'handlingcircuit2':
                return {'pos': (-280.704, -25.4946, 118.794), 'rot': None, 'rot_quat': (-0.00862686, 0.0063203, 0.98271, 0.184842)}
            elif self.road_id == 'handlingcircuit3':
                return {'pos': (-214.929, 61.2237, 118.593), 'rot': None, 'rot_quat': (-0.00947676, -0.00484788, -0.486675, 0.873518)}
            elif self.road_id == 'handlingcircuit4':
                # return {'pos': (-180.663, 117.091, 117.654), 'rot': None, 'rot_quat': (0.0227101, -0.00198367, 0.520494, 0.853561)}
                # return {'pos': (-171.183,147.699,117.438), 'rot': None, 'rot_quat': (0.001710215350613,-0.039731655269861,0.99312973022461,-0.11005393415689)}
                return {'pos': (-173.009,137.433,116.701), 'rot': None,'rot_quat': (0.0227101, -0.00198367, 0.520494, 0.853561)}
                return {'pos': (-166.679, 146.758, 117.68), 'rot': None,'rot_quat': (0.075107827782631, -0.050610285252333, 0.99587279558182, 0.0058960365131497)}
            elif self.road_id == 'rally track':
                # rally track
                return {'pos': (-374.835, 84.8178, 115.084), 'rot': None, 'rot_quat': (0, 0, 0.718422, 0.695607)}
            elif self.road_id == 'highway':
                # highway (open, farm-like)
                return {'pos': (-294.791, -255.693, 118.703), 'rot': None, 'rot_quat': (0, 0, -0.704635, 0.70957)}
            elif self.road_id == 'highwayopp':
                # highway (open, farm-like)
                return {'pos': (-542.719,-251.721,117.083), 'rot': None, 'rot_quat': (0.0098941307514906,0.0096141006797552,0.72146373987198,0.69231480360031)}
            elif self.road_id == 'default':
                # default
                return {'pos': (487.25, 178.73, 131.928), 'rot': None, 'rot_quat': (0, 0, -0.702719, 0.711467)}
        elif self.default_scenario == 'industrial':
            if self.road_id == 'west':
                # western industrial area -- didnt work with AI Driver
                return {'pos': (237.131, -379.919, 34.5561), 'rot': None, 'rot_quat': (-0.035, -0.0181, 0.949, 0.314)}
            # open industrial area -- didnt work with AI Driver
            # drift course (dirt and paved)
            elif self.road_id == 'driftcourse':
                return {'pos': (20.572, 161.438, 44.2149), 'rot': None, 'rot_quat': (-0.003, -0.005, -0.636, 0.771)}
            # rallycross course/default
            elif self.road_id == 'rallycross':
                return {'pos': (4.85287, 160.992, 44.2151), 'rot': None, 'rot_quat': (-0.0032, 0.003, 0.763, 0.646)}
            # racetrack
            elif self.road_id == 'racetrackright':
                return {'pos': (184.983, -41.0821, 42.7761), 'rot': None, 'rot_quat': (-0.005, 0.001, 0.299, 0.954)}
            elif self.road_id == 'racetrackleft':
                return {'pos': (216.578, -28.1725, 42.7788), 'rot': None, 'rot_quat': (-0.0051, -0.003147, -0.67135, 0.74112)}
            elif self.road_id == 'racetrackstartinggate':
                return {'pos':(160.905, -91.9654, 42.8511), 'rot': None, 'rot_quat':(-0.0036226876545697, 0.0065293218940496, 0.92344760894775, -0.38365218043327)}
            elif self.road_id == "racetrackstraightaway":
                return {'pos':(262.328, -35.933, 42.5965), 'rot': None, 'rot_quat':(-0.010505940765142, 0.029969356954098, -0.44812294840813, 0.89340770244598)}
            elif self.road_id == "racetrackcurves":
                return {'pos':(215.912,-243.067,45.8604), 'rot': None, 'rot_quat':(0.029027424752712,0.022241719067097,0.98601061105728,0.16262225806713)}
        elif self.default_scenario == "hirochi_raceway":
            if self.road_id == "9039" or self.road_id == "9040": # good candidate for input rect.
                #return {'pos': (292.405,-271.64,46.75), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.960669), -130)}
                return {'pos': (290.558, -277.280, 46.0), 'rot': None, 'rot_quat': self.turn_X_degrees((0, 0, -0.277698, 0.960669), -130)}
            elif self.road_id == "9205":
                return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
            elif self.road_id == "9156":
                return {'pos': (-401.98, 243.3, 25.5), 'rot': None, 'rot_quat': (0, 0, -0.277698, 0.960669)}
            else:
                return {'pos': (-453.309, 373.546, 25.3623), 'rot': None, 'rot_quat': (0, 0, -0.2777, 0.9607)}
        elif self.default_scenario == "small_island":
            if self.road_id == "int_a_small_island":
                return {"pos": (280.397, 210.259, 35.023), 'rot': None, 'rot_quat': self.turn_X_degrees((-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542), 110)}
            elif self.road_id == "ai_1":
                return {"pos": (314.573, 105.519, 37.5), 'rot': None, 'rot_quat': self.turn_X_degrees((-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542), 155)}
            else:
                return {'pos': (254.77, 233.82, 39.5792), 'rot': None, 'rot_quat': (-0.013234630227089, 0.0080483080819249, -0.00034890600363724, 0.99987995624542)}
        elif self.default_scenario == "jungle_rock_island":
            return {'pos': (-9.99082, 580.726, 156.72), 'rot': None, 'rot_quat': (-0.0067, 0.0051, 0.6231, 0.7821)}

    def create_ai_line_from_road_with_interpolation(self):
        line,points,point_colors,spheres,sphere_colors, traj = [], [], [], [], [], []
        self.road_analysis()
        print("spawn point:{}".format(self.spawn))
        print(f"{self.actual_middle[0]=}")
        print(f"{self.actual_middle[-1]=}")
        elapsed_time = 0
        for i, p in enumerate(self.actual_middle[:-1]):
            # interpolate at 1m distance
            if self.distance(p, self.actual_middle[i+1]) > 1:
                y_interp = interpolate.interp1d([p[0], self.actual_middle[i+1][0]], [p[1], self.actual_middle[i+1][1]])
                num = int(self.distance(p, self.actual_middle[i+1]))
                xs = np.linspace(p[0], self.actual_middle[i+1][0], num=num, endpoint=True)
                ys = y_interp(xs)
                for x,y in zip(xs, ys):
                    traj.append([x, y])
            else:
                elapsed_time += self.distance(p, self.actual_middle[i+1]) / 12
                traj.append([p[0],p[1]])
                linedict = {"x": p[0], "y": p[1], "z": p[2], "t": elapsed_time}
                line.append(linedict)
        # set up debug line
        for i,p in enumerate(self.actual_middle[:-1]):
            points.append([p[0], p[1], p[2]])
            point_colors.append([0, 1, 0, 0.1])
            spheres.append([p[0], p[1], p[2], 0.25])
            sphere_colors.append([1, 0, 0, 0.8])
        self.centerline_interpolated = copy.deepcopy(traj)
        self.bng.add_debug_line(points, point_colors, spheres=spheres, sphere_colors=sphere_colors, cling=True, offset=0.1)
        return line

    def ms_to_kph(self, wheelspeed):
        return wheelspeed * 3.6

    def throttle_PID(self, kph, dt):
        kp, ki, kd = 0.19, 0.0001, 0.008
        error = self.setpoint - kph
        if dt > 0:
            deriv = (error - self.prev_error) / dt
        else:
            deriv = 0
        self.integral = self.integral + error * dt
        w = kp * error + ki * self.integral + kd * deriv
        self.prev_error = error
        return w

    #return distance between two any-dimenisonal points
    def distance(self, a, b):
        sqr = sum([math.pow(ai-bi, 2) for ai, bi in zip(a,b)])
        # return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
        return math.sqrt(sqr)

    def road_analysis(self):
        print("Performing road analysis...")
        # self.get_nearby_racetrack_roads(point_of_in=(-391.0,-798.8, 139.7))
        # self.get_nearby_racetrack_roads(point_of_in=(57.04786682128906, -150.53302001953125, 125.5))
        # self.plot_racetrack_roads()

        print(f"Getting road {self.road_id}...")
        edges = self.bng.get_road_edges(self.road_id)
        if self.reverse:
            edges.reverse()
            print(f"new spawn={edges[0]['middle']}")
        else:
            print(f"reversed spawn={edges[-1]['middle']}")
        self.actual_middle = [edge['middle'] for edge in edges]
        self.roadleft = [edge['left'] for edge in edges]
        self.roadright = [edge['right'] for edge in edges]
        self.adjusted_middle = [edge['middle'] for edge in edges]
        self.centerline = self.actual_middle

    def plot_racetrack_roads(self):
        roads = self.bng.get_roads()
        print("got roads")
        sp = self.spawn_point()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        symbs = ['-', '--', '-.', ':', '.', ',', 'v', 'o', '1', ]
        for road in roads:
            road_edges = self.bng.get_road_edges(road)
            if len(road_edges) < 100:
                continue
            x_temp, y_temp = [], []
            xy_def = [edge['middle'][:2] for edge in road_edges]
            dists = [self.distance(xy_def[i], xy_def[i + 1]) for i, p in enumerate(xy_def[:-1])]
            s = sum(dists)

            if (s < 500) or s > 800:
                continue
            for edge in road_edges:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
            print(f"{road}\tdist={s:.1f}\t{len(road_edges)=}\tstart={road_edges[0]['middle']}")
            symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
        plt.plot([sp['pos'][0]], [sp['pos'][1]], "bo")
        plt.title(self.default_scenario)
        plt.legend(ncol=10)
        plt.show()
        plt.pause(0.001)

    def get_nearby_racetrack_roads(self, point_of_in):
        print(f"Plotting nearby roads to point={point_of_in}")
        roads = self.bng.get_roads()
        print("retrieved roads")
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        symbs = ['-', '--', '-.', ':', '.', ',', 'v', 'o', '1', ]
        for road in roads:
            road_edges = self.bng.get_road_edges(road)
            x_temp, y_temp = [], []
            if len(road_edges) < 100:
                continue
            xy_def = [edge['middle'][:2] for edge in road_edges]
            # dists = [self.distance(xy_def[i], xy_def[i + 1]) for i, p in enumerate(xy_def[:-1:5])]
            # road_len = sum(dists)
            dists = [self.distance(i, point_of_in) for i in xy_def]
            s = min(dists)
            if (s > 100): # or road_len < 200:
                continue
            for edge in road_edges:
                x_temp.append(edge['middle'][0])
                y_temp.append(edge['middle'][1])
            symb = '{}{}'.format(random.choice(colors), random.choice(symbs))
            plt.plot(x_temp, y_temp, symb, label=road)
            print(f"{road=}\tstart=({x_temp[0]},{y_temp[0]},{road_edges[0]['middle'][2]})\t{road_edges[0]['middle']}")
        plt.plot([point_of_in[0]], [point_of_in[1]], "bo")
        plt.title(f"{self.default_scenario} poi={point_of_in}")
        plt.legend(ncol=10)
        plt.show()
        plt.pause(0.001)

    def turn_X_degrees(self, rot_quat, degrees=90):
        r = R.from_quat(list(rot_quat))
        r = r.as_euler('xyz', degrees=True)
        r[2] = r[2] + degrees
        r = R.from_euler('xyz', r, degrees=True)
        return tuple(r.as_quat())

    def dist_from_line(self, centerline, point):
        a = [x[0:2] for x in centerline[:-1]]
        b = [x[0:2] for x in centerline[1:]]
        # a = np.array(a)
        # b = np.array(b)
        # print(f"{a.shape=}\t{b.shape=}")
        dist = self.lineseg_dists(point[0:2], np.array(a), np.array(b))
        return dist

    def lineseg_dists(self, p, a, b):
        """Cartesian distance from point to line segment
        Edited to support arguments as series, from:
        https://stackoverflow.com/a/54442561/11208892
        Args:
            - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
            - a: np.array of shape (x, 2)
            - b: np.array of shape (x, 2)
        """
        # normalized tangent vectors
        d_ba = b - a
        d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

        # signed parallel distance components
        # rowwise dot products of 2D vectors
        s = np.multiply(a - p, d).sum(axis=1)
        t = np.multiply(p - b, d).sum(axis=1)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(s))])

        # perpendicular distance component
        # rowwise cross products of 2D vectors
        d_pa = p - a
        c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

        return np.hypot(h, c)

    def plot_deviation(self, model, deflation_pattern, start_viz=False):
        plt.figure(20, dpi=180)
        plt.clf()
        i = 0; x = []; y = []
        for point in self.centerline:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, "b-")
        x, y = [], []
        for point in self.roadleft:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, "b-")
        x, y = [], []
        for point in self.roadright:
            x.append(point[0])
            y.append(point[1])
        plt.plot(x, y, "b-", label="Road")
        for t in self.trajectories:
            x, y = [], []
            for point in t:
                x.append(point[0])
                y.append(point[1])
            plt.plot(x, y) #, label="Run {}".format(i))
            i += 1
        plt.title(f'Trajectories with {model}\n{deflation_pattern.split("/")[-1]}')
        plt.legend(fontsize=8)
        # if start_viz:
        #     plt.xlim([200,400])
        #     plt.ylim([-300,-100])
        # else:
        #     plt.xlim([-350, 650])
        #     plt.ylim([-325, 475])
        plt.draw()
        plt.savefig(f"{deflation_pattern}-ep{self.episode}-{model}-trajs-so-far.jpg")
        plt.clf()

    def get_distance_traveled(self, traj):
        dist = 0.0
        for i in range(len(traj[:-1])):
            dist += math.sqrt(
                math.pow(traj[i][0] - traj[i + 1][0], 2) + math.pow(traj[i][1] - traj[i + 1][1], 2) + math.pow(
                    traj[i][2] - traj[i + 1][2], 2))
        return dist

    def plot_durations(self, rewards, save=False, title="temp"):
        # plt.figure(2)
        plt.figure(4, figsize=(10,10), dpi=100)
        plt.clf()
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        # durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        # plt.ylabel('Duration')
        plt.plot(rewards_t.numpy(), label="Rewards")
        # plt.plot(durations_t.numpy(), "--")
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 20:
            means = rewards_t.unfold(0, 20, 1).mean(1).view(-1)
            means = torch.cat((torch.ones(19) * means[0], means))
            plt.plot(means.numpy(), '--')
        plt.legend()
        # plt.pause(0.001)  # pause a bit so that plots are updated
        # if is_ipython:
        #     display.clear_output(wait=True)
        #     display.display(plt.gcf())

        if save:
            plt.savefig(f"{title}-training_performance.jpg")

    def add_qr_cubes(self, scenario, qrbox_filename='posefiles/qr_box_locations.txt'):
        global qr_positions
        qr_positions = []
        with open(qrbox_filename, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split(' ')
                pos = line[0].split(',')
                pos = tuple([float(i) for i in pos])
                rot_quat = line[1].split(',')
                rot_quat = tuple([float(j) for j in rot_quat])
                qr_positions.append([copy.deepcopy(pos), copy.deepcopy(rot_quat)])
                box = ScenarioObject(oid='qrbox_{}'.format(i), name='qrbox2', otype='BeamNGVehicle', pos=pos, rot=None,
                                     rot_quat=rot_quat, scale=(1, 1, 1), JBeam='qrbox2', datablock="default_vehicle")
                scenario.add_object(box)

    # with warp
    def overlay_transparent(self, img1, img2, corners):
        import kornia
        print(f"{img1.shape=}") # img1.shape=(135, 240, 3)
        print(f"{img2.shape=}") # img2.shape=(1, 1)
        orig = torch.from_numpy(img1[None]).permute(0, 3, 1, 2) / 255.0
        pert = torch.from_numpy(img2).permute(0, 3, 1, 2) / 255.0 # determined by observation space high and low

        _, c, h, w = _, *pert_shape = pert.shape
        _, *orig_shape = orig.shape
        patch_coords = corners[None]
        src_coords = np.tile(
            np.array(
                [
                    [
                        [0.0, 0.0],
                        [w - 1.0, 0.0],
                        [0.0, h - 1.0],
                        [w - 1.0, h - 1.0],
                    ]
                ]
            ),
            (len(patch_coords), 1, 1),
        )
        src_coords = torch.from_numpy(src_coords).float()
        patch_coords = torch.from_numpy(patch_coords).float()

        # build the transforms to and from image patches
        try:
            perspective_transforms = kornia.geometry.transform.get_perspective_transform(src_coords, patch_coords)
        except Exception as e:
            print(f"{e=}")
            print(f"{src_coords=}")
            print(f"{patch_coords=}")

        perturbation_warp = kornia.geometry.transform.warp_perspective(
            pert,
            perspective_transforms,
            dsize=orig_shape[1:],
            mode="nearest",
            align_corners=True
        )
        mask_patch = torch.ones(1, *pert_shape)
        warp_masks = kornia.geometry.transform.warp_perspective(
            mask_patch, perspective_transforms, dsize=orig_shape[1:],
            mode="nearest",
            align_corners=True
        )
        perturbed_img = orig * (1 - warp_masks)
        perturbed_img += perturbation_warp * warp_masks
        return (perturbed_img.permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    # uses contour detection
    # @ignore_warnings
    def get_qr_corners_from_colorseg_image(self, image):
        from skimage import util
        image = np.array(image)
        # cv2.imshow('colorseg', image)
        # cv2.waitKey(1)
        # hsv mask image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        light_color = (50, 230, 0)  # (50, 235, 235) #(0, 200, 0)
        dark_color = (90, 256, 256)  # (70, 256, 256) #(169, 256, 256)
        mask = cv2.inRange(hsv_image, light_color, dark_color)
        image = cv2.bitwise_and(image, image, mask=mask)

        # convert image to inverted greyscale
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
        inverted_img = util.invert(imgGray)
        inverted_img = np.uint8(inverted_img)
        inverted_img = 255 - inverted_img
        inverted_img = cv2.GaussianBlur(inverted_img, (3, 3), 0)  # 9

        # contour detection
        ret, thresh = cv2.threshold(inverted_img, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        if contours == [] or np.array(contours).shape[0] < 2:
            return [[[0, 0], [0, 0], [0, 0], [0, 0]]], None
        else:
            epsilon = 0.1 * cv2.arcLength(np.float32(contours[1]), True)
            approx = cv2.approxPolyDP(np.float32(contours[1]), epsilon, True)

            contours = np.array([c[0] for c in contours[1]])
            approx = [c[0] for c in approx]
            if len(approx) < 4:
                return [[[0, 0], [0, 0], [0, 0], [0, 0]]], None

            def sortClockwise(approx):
                xs = [a[0] for a in approx]
                ys = [a[1] for a in approx]
                center = [int(sum(xs) / len(xs)), int(sum(ys) / len(ys))]

                def sortFxnX(e):
                    return e[0]

                def sortFxnY(e):
                    return e[1]

                approx = list(approx)
                approx.sort(key=sortFxnX)
                midpt = int(len(approx) / 2)
                leftedge = list(approx[:midpt])
                rightedge = list(approx[midpt:])
                leftedge.sort(key=sortFxnY)
                rightedge.sort(key=sortFxnY)
                approx = [leftedge[0], leftedge[1], rightedge[1], rightedge[0]]
                return approx, leftedge, rightedge, center

            approx, le, re, center = sortClockwise(approx)
            for i, c in enumerate(le):
                cv2.circle(image, tuple([int(x) for x in c]), radius=1, color=(100 + i * 20, 0, 0), thickness=2)  # blue
            for i, c in enumerate(re):
                cv2.circle(image, tuple([int(x) for x in c]), radius=1, color=(0, 0, 100 + i * 20), thickness=2)  # blue
            cv2.circle(image, tuple(center), radius=1, color=(203, 192, 255), thickness=2)  # lite pink
            if len(approx) > 3:
                cv2.circle(image, tuple([int(x) for x in approx[0]]), radius=1, color=(0, 255, 0), thickness=2)  # green
                cv2.circle(image, tuple([int(x) for x in approx[2]]), radius=1, color=(0, 0, 255), thickness=2)  # red
                cv2.circle(image, tuple([int(x) for x in approx[3]]), radius=1, color=(255, 255, 255),
                           thickness=2)  # white
                cv2.circle(image, tuple([int(x) for x in approx[1]]), radius=1, color=(147, 20, 255),
                           thickness=2)  # pink

            keypoints = [[tuple(approx[0]), tuple(approx[3]),
                          tuple(approx[1]), tuple(approx[2])]]
            return keypoints, image