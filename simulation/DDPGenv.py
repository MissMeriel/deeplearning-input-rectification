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
                 base_model=None):
        super(CarEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,1))
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
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self.viewer = None

        self.start_ts = 0
        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
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
        self.car_state = sensors #self.getCarState(sensors)
        self.state["pose"] = self.vehicle.state["pos"]
        self.state["collision"] = sensors["damage"]["damage"] > 0
        image = np.array(sensors['front_cam']['colour'].convert('RGB'), dtype=np.uint8)
        if self.image_shape[0] == 1:
            image = self.rgb2gray(image)
        image = image.reshape(self.image_shape)
        return image

    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return np.array(gray, dtype=np.uint8)

    def step(self, action):
        sensors = self.bng.poll_sensors(self.vehicle)
        kph = self.ms_to_kph(sensors['electrics']['wheelspeed'])
        dt = (sensors['timer']['time'] - self.start_time) - self.runtime
        self.runtime = sensors['timer']['time'] - self.start_time
        throttle = self.throttle_PID(kph, dt)
        image = np.array(sensors['front_cam']['colour'].convert('RGB'), dtype=np.uint8)
        transform = T.Compose([T.ToTensor()])
        features = transform(image)[None]
        base_model_inf = self.base_model(features).item()
        self.base_model_inf.append(base_model_inf)

        # while self.evaluator() == 1:
        while True:
            print(f"\t{base_model_inf=:.3f}\t RL_action={action.item():.3f}")

            outside_track, distance_from_center, leftrightcenter, segment_shape = self.has_car_left_track()
            done = outside_track or self.state["collision"]

            if done:
                obs = self._get_obs()
                reward = math.pow(2, 4.0 - distance_from_center)
                return obs, reward, done, self.state

            if abs(base_model_inf) > 0.2:
                self.setpoint = 30
            else:
                self.setpoint = 40

            self.vehicle.control(throttle=throttle, steering=base_model_inf, brake=0.0)
            self.bng.step(1, wait=True)

            self.actions.append(action)

            sensors = self.bng.poll_sensors(self.vehicle)
            kph = self.ms_to_kph(sensors['electrics']['wheelspeed'])
            dt = (sensors['timer']['time'] - self.start_time) - self.runtime
            self.runtime = sensors['timer']['time'] - self.start_time
            throttle = self.throttle_PID(kph, dt)
            image = np.array(sensors['front_cam']['colour'].convert('RGB'), dtype=np.uint8)
            transform = T.Compose([T.ToTensor()])
            features = transform(image)[None]
            base_model_inf = self.base_model(features).item()
            self.base_model_inf.append(base_model_inf)

        if abs(action) > 0.2:
            self.setpoint = 30
        else:
            self.setpoint = 40

        self.vehicle.control(throttle=throttle, steering=action.item(), brake=0.0)
        self.bng.step(1, wait=True)
        obs = self._get_obs()
        outside_track, distance_from_center, leftrightcenter, segment_shape = self.has_car_left_track()
        reward = math.pow(2, 4.0 - distance_from_center)
        done = outside_track or self.state["collision"]
        self.current_rewards.append(reward)
        print(f"STEP() \n\taction={action.item():.3f}\t{base_model_inf=:.2f}\n\t{reward=:.1f}\n\t{done=}\t{outside_track=}\tcollision={self.state['collision']}")
        return obs, reward, done, self.state

    def reset(self):
        print(f"RESET()")
        self.trajectories.append(self.current_trajectory)
        self.all_rewards.append(sum(self.current_rewards))
        dist = self.get_distance_traveled(self.current_trajectory)
        # self.plot_deviation(f"{self.model} {dist=:.1f} ep={self.episode} start", self.deflation_pattern, start_viz=True)
        self.plot_deviation(f"{self.model} {dist=:.1f} ep={self.episode}", self.deflation_pattern, start_viz=False)
        self.plot_durations(self.all_rewards, save=True, title=self.deflation_pattern)
        self.episode += 1
        self.current_trajectory = []
        self.current_rewards = []
        self.integral, self.prev_error = 0.0, 0.0
        self.bng.restart_scenario()
        self.bng.step(1, wait=True)
        self.vehicle.update_vehicle()
        sensors = self.bng.poll_sensors(self.vehicle)
        spawnpoint = [290.558, -277.28, 46.0]
        endpoint = [-346.5, 431.0, 30.8]  # centerline[-1]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.prev_error = self.setpoint
        self.overall_damage, self.runtime, self.wheelspeed, self.distance_from_center = 0.0, 0.0, 0.0, 0.0
        self.total_loops, self.total_imgs, self.total_predictions = 0, 0, 0
        self.start_time = sensors['timer']['time']
        self.image = np.array(sensors['front_cam']['colour'].convert('RGB'))
        # image = np.array(sensors['front_cam']['colour'].convert('RGB'))
        # self.image = self.rgb2gray(self.image).reshape(self.image_shape)
        if self.image_shape[0] == 1:
            self.image = self.rgb2gray(self.image)
        self.image = self.image.reshape(self.image_shape)
        self.outside_track = False
        self.done = False
        self.action_inputs = [-1, 0, 1]
        self.action_indices = [0, 1, 2]
        self.states, self.actions, self.probs, self.rewards, self.critic_values = [], [], [], [], []
        self.traj = []
        return self._get_obs()

    def render(self):
        return self._get_obs()

    # bad = 0, good = 1
    def evaluator(self, steer=None):
        outside_track, distance_from_center, leftrightcenter, segment_shape = self.has_car_left_track()
        if steer is None:
            steer = self.base_model_inf[-1]
        if leftrightcenter == 0 and abs(steer) < 1.1 and segment_shape == 0:
            # centered, driving straight, straight road
            print(f"EVAL(base_model_inf={steer:.3f})=GOOD \n\tcentered, driving straight, straight road")
            return 1
        elif leftrightcenter == 0 and steer < -0.05 and segment_shape == 1:
            # centered, driving left, left curve road
            print(f"EVAL(base_model_inf={steer:.3f})=GOOD\n\tcentered, driving left, left curve road")
            return 1
        elif leftrightcenter == 0 and steer > 0.05 and segment_shape == 2:
            # centered, driving right, right curve road
            print(f"EVAL(base_model_inf={steer:.3f})=GOOD\n\tcentered, driving right, right curve road")
            return 1
        elif leftrightcenter == 1 and steer > 0.05:
            # left of center, turning right
            print(f"EVAL(base_model_inf={steer:.3f})=GOOD\n\tleft of center, turning right")
            return 1
        elif leftrightcenter == 2 and steer < -0.05:
            # right of center, turning left
            print(f"EVAL(base_model_inf={steer:.3f})=GOOD\n\tright of center, turning left")
            return 1
        else:
            print(f"EVAL(base_model_inf={steer:.3f})=BAD")
            return 0


    ################################# BEAMNG HELPERS #################################

    def setup_sensors(self):
        camera_pos = (-0.5, 0.38, 1.3)
        camera_dir = (0, 1.0, 0)
        fov = 51  # 60 works for full lap #63 breaks on hairpin turn
        resolution = (self.image_shape[2], self.image_shape[1])  # (400,225) #(320, 180) #(1280,960) #(512, 512)
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
        for i,p in enumerate(self.actual_middle[:-1]):
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
        kp = 0.19;
        ki = 0.0001;
        kd = 0.008
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


    # track ~12.50m wide; car ~1.85m wide
    def has_car_left_track(self):
        self.vehicle.update_vehicle()
        vehicle_pos = self.vehicle.state['pos']
        vehicle_bbox = self.vehicle.get_bbox()
        distance_from_centerline = self.dist_from_line(self.centerline_interpolated, vehicle_pos)
        dist = min(distance_from_centerline)
        i = np.where(distance_from_centerline == dist)[0][0]
        leftrightcenter = self.get_position_relative_to_centerline(dist, i, centerdist=1)
        segment_shape = self.get_current_segment_shape(vehicle_pos)
        return dist > 4.0, dist, leftrightcenter, segment_shape

    def get_current_segment_shape(self, vehicle_pos):
        distance_from_centerline = self.dist_from_line(self.actual_middle, vehicle_pos)
        dist = min(distance_from_centerline)
        i = np.where(distance_from_centerline == dist)[0][0]
        # print(f"{i=}\t{vehicle_pos=}")
        A = np.array(self.actual_middle[(i + 1) % len(self.actual_middle)])
        B = np.array(self.actual_middle[i])
        C = np.array(self.roadright[i])
        try:
            theta = math.acos(np.vdot(B-A, B-C) / np.linalg.norm(B-A) * np.linalg.norm(B-C))
        except ValueError:
            print(f"{A=}\t{B=}\t{C=}")
            theta = 0
        theta_deg = math.degrees(theta)
        # print(f"{math.degrees(theta)=:.1f}")
        if theta_deg > 100:
            # print(f"Road curving left \t{theta_deg=:.1f}")
            return 1
        elif theta_deg < 80:
            # print(f"Road curving right \t{theta_deg=:.1f}")
            return 2
        else:
            # print(f"Road is straight \t{theta_deg=:.1f}")
            return 0

    def get_position_relative_to_centerline(self, dist, i, centerdist=1):
        A = self.centerline_interpolated[i]
        B = self.centerline_interpolated[(i + 1) % len(self.centerline_interpolated)]
        P = self.vehicle.state['pos']
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


    def dist_from_line(self, centerline, point):
        a = [[x[0],x[1]] for x in centerline[:-1]]
        b = [[x[0],x[1]] for x in centerline[1:]]
        a = np.array(a)
        b = np.array(b)
        dist = self.lineseg_dists([point[0], point[1]], a, b)
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
        # global centerline, roadleft, roadright
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
            x = []; y = []
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
        plt.figure(4, figsize=(3, 3), dpi=100)
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