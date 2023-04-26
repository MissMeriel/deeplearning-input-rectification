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
import pickle
import matplotlib
from utils import *
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


class CarEnv(gym.Env):

    # metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape=(3, 135, 240), obs_shape=(3, 135, 240), model="DQN", filepathroot=".", beamngpath="C:/Users/Meriel/Documents",
                    beamnginstance="BeamNG.research", port=64356, scenario="west_coast_usa", road_id="12146", reverse=False,
                    base_model=None, test_model=False, seg=None, transf=None, topo="straight", eval_eps=0.05):
        super(CarEnv, self).__init__()

        self.metadata = {"render.modes":
            ["human",
            "rgb_array",
            "depth_array",
        ]}
        self.start_time = time.time()
        self.transf = transf
        self.topo = topo
        self.eval_eps=eval_eps
        self.test_eps=eval_eps
        self.f = None
        self.ep_dir = None
        self.seg = seg
        self.test_model = test_model
        self.transform = T.Compose([T.ToTensor()])
        self.episode_steps, self.frames_adjusted = 0, 0
        self.steer_integral, self.steer_prev_error = 0., 0.
        self.beamngpath = beamngpath
        if base_model is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.base_model = torch.load(base_model, map_location=device).eval() #torch.load(base_model).to(device)
        self.default_scenario = scenario
        self.road_id = road_id
        self.reverse = reverse
        self.cutoff_point = get_cutoff_point(self.default_scenario, self.road_id, seg)
        self.integral, self.prev_error = 0.0, 0.0
        self.steer_integral, self.steer_prev_error = 0.0, 0.0
        self.overall_throttle_setpoint = 40
        self.setpoint = self.overall_throttle_setpoint
        self.lanewidth = 3.75  # 2.25
        self.track_cam_pos = None
        self.centerline, self.centerline_interpolated, self.roadleft, self.roadright = [], [], [], []
        self.trajectories, self.current_trajectory = [], []
        self.all_step_counts, self.all_rewards, self.current_rewards, self.all_intervention_rates = [], [], [], []
        self.actions, self.base_model_inf = [], []
        self.image_shape = image_shape
        self.steps_per_sec = 15
        self.runtime = 0.0
        self.episode = -1
        random.seed(1703)
        self.model = model
        self.filepathroot = filepathroot
        beamng = BeamNGpy('localhost', port=port, home=f'{self.beamngpath}/BeamNG.research.v1.7.0.1', user=f'{self.beamngpath}/{beamnginstance}')

        self.scenario = Scenario(self.default_scenario, 'RL_Agent-train')
        self.vehicle = Vehicle('ego_vehicle', model="hopper", licence='EGO', color="green")
        # self.obs_shape = obs_shape

        self.vehicle = setup_sensors(transf, image_shape, self.vehicle)
        self.spawn = spawn_point(self.default_scenario, road_id, seg, reverse)
        self.scenario.add_vehicle(self.vehicle, pos=self.spawn['pos'], rot=None, rot_quat=self.spawn['rot_quat'])
        self.scenario.make(beamng)
        self.bng = beamng.open(launch=True)
        self.bng.set_deterministic()
        self.bng.set_steps_per_second(self.steps_per_sec)
        self.bng.load_scenario(self.scenario)
        self.bng, self.centerline_interpolated, self.centerline, self.roadleft, self.roadright = create_ai_line_from_road_with_interpolation(self.bng, self.road_id)
        self.bng.start_scenario()
        self.bng.pause()
        assert self.vehicle.skt
        ###########################################################################
        features_shape = self.base_model.features(torch.rand((self.image_shape))[None]).detach().numpy()
        # print(f"{features_shape.shape=}")
        # print(f"{features_shape.dtype=}")
        self.obs_shape = tuple(np.array(features_shape.shape))
        # print(f"{self.obs_shape=}")

        self.observation_space = spaces.Box(float('-inf') , float('inf') , shape=self.obs_shape, dtype=np.float32)
        self.viewer = None

        self.start_ts = 0
        self.state = {
            # "position": np.zeros(3),
            # "prev_position": np.zeros(3),
            # "pose": None,
            # "prev_pose": None,
            # "collision": False,
        }
        self.action_space = spaces.Discrete(7)
        self.image = None
        self.car_state = None

    def _do_action(self, action):
        sensors = self.bng.poll_sensors(self.vehicle)
        kph = ms_to_kph(sensors['electrics']['wheelspeed'])
        dt = (sensors['timer']['time'] - self.start_time) - self.runtime
        self.runtime = sensors['timer']['time'] - self.start_time
        throttle = self.throttle_PID(kph, dt)
        if action == 0:
            steer = 0
        elif action == 1:
            steer = 0.5
        elif action == 2:
            steer = -0.5
        elif action == 3:
            steer = 0.25
        elif action == 4:
            steer = -0.25
        elif action == 5:
            steer = 1.0
        else:
            steer = -1.0

        if abs(steer) > 0.2:
            self.setpoint = 30
        else:
            self.setpoint = 40

        self.vehicle.control(throttle=throttle, steering=steer, brake=0.0)
        self.bng.step(1, wait=True)


    def _get_obs(self):
        # self.state["prev_pose"] = self.vehicle.state["pos"]
        self.vehicle.update_vehicle()
        sensors = self.bng.poll_sensors(self.vehicle)
        self.current_trajectory.append(self.vehicle.state["pos"])
        self.car_state = sensors
        # self.state["pose"] = self.vehicle.state["pos"]
        # self.state["collision"] = sensors["damage"]["damage"] > 0
        image = np.array(sensors['front_cam']['colour'].convert('RGB'), dtype=np.uint8)
        if len(self.obs_shape) == 3 and self.obs_shape[0] == 1:
            image = rgb2gray(image)
        if len(self.obs_shape) == 3:
            image = image.reshape(self.obs_shape)
            return image
        else:
            # print(f"{image.shape=}")
            torchimg = self.transform(image)[None]
            # print(f"{torchimg.shape=}")
            imgfeatures = self.base_model.features(torchimg).detach().numpy()
            imgfeatures = imgfeatures.reshape(self.obs_shape)
            # print(f"{imgfeatures.shape=}")
            # print(f"{type(imgfeatures)=}")
            # print(f"{imgfeatures.dtype=}")
            return imgfeatures


    # track ~12.50m wide; car ~1.85m wide
    def has_car_left_track(self):
        self.vehicle.update_vehicle()
        vehicle_pos = self.vehicle.state['front']
        distance_from_centerline = dist_from_line(self.centerline_interpolated, vehicle_pos)
        dist = min(distance_from_centerline)
        return dist > 4.0, dist


    def step(self, action):
        self.actions.append(action)
        self._do_action(action)
        obs = self._get_obs()
        outside_track, distance_from_center = self.has_car_left_track()
        dist_to_cutoff = distance(self.vehicle.state["pos"][:2], self.cutoff_point[:2])
        if outside_track:
            reward = -500
        elif dist_to_cutoff < 12:
            reward = 500
        else:
            reward = math.pow(2, 4.0 - distance_from_center)
        done = outside_track or self.car_state["damage"]["damage"] > 1 or dist_to_cutoff < 12
        self.current_rewards.append(reward)
        # print(f"STEP() {action=}\t{reward=:.2f}\t{done=}\t{outside_track=}\t{self.state['collision']=}")
        self.episode_steps += 1
        return obs, reward, done, self.state

    def reset(self):
        print(f"RESET() {len(self.current_trajectory)=}")
        self.trajectories.append(self.current_trajectory)
        self.all_rewards.append(sum(self.current_rewards))
        self.all_step_counts.append(self.episode_steps)
        dist = get_distance_traveled(self.current_trajectory)
        dist_from_centerline = []
        if self.episode > -1 and len(self.current_trajectory) > 0 and self.episode_steps > 0:
            for i in self.current_trajectory:
                distance_from_centerline = dist_from_line(self.centerline_interpolated, i)
                dist_from_centerline.append(min(distance_from_centerline))
            plot_deviation(f"{self.model} {dist=:.1f} ep={self.episode}",
                            self.filepathroot + str(
                            f" avg dist ctr:{sum(dist_from_centerline) / len(dist_from_centerline):.3f} frames adj:{self.frames_adjusted / self.episode_steps:.3f}  rew={sum(self.current_rewards):.1f}"),
                            self.centerline, self.roadleft, self.roadright, self.trajectories, self.topo,
                            save_path=f"{self.filepathroot}/trajs-ep{self.episode}.jpg", start_viz=False)
            plot_durations(self.all_rewards, save=True, title=self.filepathroot)
            plot_rewards_by_steps(self.all_rewards, self.all_step_counts, save=True, title=self.filepathroot)
            summary = {
                "episode" : self.episode,
                "rewards": self.current_rewards,
                "trajectory": self.current_trajectory,
                "total_steps": self.episode_steps,
                "frames_adjusted_count": self.frames_adjusted,
                "dist_from_centerline": dist_from_centerline,
                "image_shape":self.image_shape,
                "obs_shape": self.obs_shape,
                "action_space": self.action_space,
                "wall_clock_time": time.time() - self.start_time,
                "sim_time": self.runtime
            }
            picklefile = open(f"{self.filepathroot}/summary-epi{self.episode:03d}.pickle", 'wb')
            pickle.dump(summary, picklefile)
            picklefile.close()
        self.episode += 1
        self.current_trajectory = []
        self.current_rewards = []
        self.integral, self.prev_error = 0.0, 0.0
        self.bng.restart_scenario()
        self.bng.step(1, wait=True)
        self.vehicle.update_vehicle()
        sensors = self.bng.poll_sensors(self.vehicle)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.prev_error = self.setpoint
        self.episode_steps = 0
        self.runtime = 0.
        self.start_time = sensors['timer']['time']
        self.image = np.array(sensors['front_cam']['colour'].convert('RGB'))
        if self.obs_shape[0] == 1:
            self.image = rgb2gray(self.image)
        self.outside_track = False
        self.done = False
        self.states, self.actions, self.probs, self.rewards, self.critic_values = [], [], [], [], []
        return self._get_obs()

    def render(self):
        return self._get_obs()

    def close(self):
        # self.f.close()
        self.bng.close()

    def get_progress(self):
        dist = get_distance_traveled(self.current_trajectory)
        dist_from_centerline = []
        for i in self.current_trajectory:
            distance_from_centerline = dist_from_line(self.centerline_interpolated, i)
            dist_from_centerline.append(min(distance_from_centerline))

        summary = {
            "episode": self.episode,
            "rewards": self.current_rewards,
            "trajectory": self.current_trajectory,
            "total_steps": self.episode_steps,
            "frames_adjusted_count": self.frames_adjusted,
            "dist_from_centerline": dist_from_centerline,
            "image_shape": self.image_shape,
            "obs_shape": self.obs_shape,
            "action_space": self.action_space,
            "dist_travelled": dist
        }
        return summary

    ################################# BEAMNG HELPERS #################################


    def steering_PID(self, curr_steering, steer_setpoint, dt):
        if dt == 0:
            return 0
        if "winding" in self.topo:
            kp = 0.425; ki = 0.00; kd = 0.0  # using LRC and dist to ctrline; Average deviation: 1.023
        elif "straight" in self.topo:
            # kp = 0.8125; ki = 0.00; kd = 0.2
            kp = 0.1; ki = 0.00; kd = 0.01  # decent on straight Average deviation: 1.096
        elif "Rturn" in self.topo:
            # kp = 0.8125; ki = 0.00; kd = 0.0 #0.3
            kp = 0.325; ki = 0.00; kd = 0.0 #0.3
        elif "Lturn" in self.topo:
            kp = 0.5; ki = 0.00; kd = 0.3
        else:
            kp = 0.75; ki = 0.01; kd = 0.2  # decent
        error = steer_setpoint - curr_steering
        deriv = (error - self.steer_prev_error) / dt
        self.steer_integral = self.steer_integral + error * dt
        w = kp * error + ki * self.steer_integral + kd * deriv
        # print(f"steering_PID({curr_steering=:.3f}  \t{steer_setpoint=:.3f}  \t{dt=:.3f})  \t{self.steer_prev_error=:.3f}  "
        #       f"\n{error:.3f} \t{deriv=:.3f} \t{w=:.3f}")
        self.steer_prev_error = error
        return w

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